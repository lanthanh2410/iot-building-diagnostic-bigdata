"""Train-fitted, zone-specific IoT feature engineering for Pandas and Spark.

Fit on the training period only; transform both training and later test records
with the same medians. Neither the target nor identifiers enter the features.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from xml.sax.saxutils import escape

MISSING_COLUMNS = (
    "occupancy_count", "tvoc_ppb", "vibration_mm_s", "window_open_pct"
)
PHYSICAL_INPUT_COLUMNS = (
    "indoor_temp_c", "outdoor_temp_c", "indoor_humidity_pct",
    "outdoor_humidity_pct", "pm25_ug_m3", "outdoor_pm25_ug_m3",
    "filter_pressure_pa", "air_flow_m3_h", "hvac_power_kw",
)
ENGINEERED_COLUMNS = (
    "delta_temp_c", "delta_humidity_pct", "pm25_indoor_outdoor_ratio",
    "filter_pressure_per_airflow", "hvac_kw_per_airflow",
)
LABELS = ("normal", "ventilation_issue", "thermal_issue")
MODEL_FEATURE_COLUMNS = (
    "floor_number", *MISSING_COLUMNS, *PHYSICAL_INPUT_COLUMNS,
    "co2_ppm", "fan_speed_rpm", "maintenance_days", "equipment_age_years",
    *ENGINEERED_COLUMNS,
)


def _require_columns(columns, required):
    absent = sorted(set(required) - set(columns))
    if absent:
        raise ValueError("Missing required columns: " + ", ".join(absent))


class PandasZoneMedianFeatureEngineer:
    """Sklearn-compatible fit/transform semantics without requiring sklearn.

    Missing zones and zones with an entirely missing column use the training
    population's global median for that column.
    """

    def fit(self, train):
        import numpy as np
        import pandas as pd

        _require_columns(train.columns, ("building_zone", *MISSING_COLUMNS))
        if train.empty:
            raise ValueError("Training data is empty")
        if train["building_zone"].isna().any():
            raise ValueError("Training building_zone contains null values")
        values = train.loc[:, MISSING_COLUMNS].apply(
            pd.to_numeric, errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        self.global_medians_ = values.median().to_dict()
        if any(pd.isna(value) for value in self.global_medians_.values()):
            raise ValueError("A column has no observed training values")
        self.zone_medians_ = values.groupby(train["building_zone"]).median()
        return self

    def transform(self, frame):
        import numpy as np
        import pandas as pd

        if not hasattr(self, "zone_medians_"):
            raise RuntimeError("Call fit(train) before transform(frame)")
        _require_columns(frame.columns, ("building_zone", *MISSING_COLUMNS,
                                         *PHYSICAL_INPUT_COLUMNS))
        if frame["building_zone"].isna().any():
            raise ValueError("building_zone contains null values")
        result = frame.copy()
        for column in (*MISSING_COLUMNS, *PHYSICAL_INPUT_COLUMNS):
            result[column] = pd.to_numeric(result[column], errors="coerce")
            result[column] = result[column].replace([np.inf, -np.inf], np.nan)
        for column in MISSING_COLUMNS:
            zone_values = result["building_zone"].map(self.zone_medians_[column])
            result[column] = result[column].fillna(zone_values).fillna(
                self.global_medians_[column]
            )
        result["delta_temp_c"] = result["indoor_temp_c"] - result["outdoor_temp_c"]
        result["delta_humidity_pct"] = (
            result["indoor_humidity_pct"] - result["outdoor_humidity_pct"]
        )
        for output, numerator, denominator in (
            ("pm25_indoor_outdoor_ratio", "pm25_ug_m3", "outdoor_pm25_ug_m3"),
            ("filter_pressure_per_airflow", "filter_pressure_pa", "air_flow_m3_h"),
            ("hvac_kw_per_airflow", "hvac_power_kw", "air_flow_m3_h"),
        ):
            result[output] = result[numerator].div(
                result[denominator].where(result[denominator] > 0)
            ).replace([np.inf, -np.inf], np.nan)
        return result

    def fit_transform(self, train):
        return self.fit(train).transform(train)


class SparkZoneMedianFeatureEngineer:
    """Fit Spark ML Imputer median models per zone on training data only.

    Each zone is fitted independently because Spark's Imputer has no groupBy
    parameter. At transform time, a broadcast table of fitted surrogates keeps
    prediction to one distributed join, including for previously unseen zones.
    """

    def fit(self, train):
        from pyspark.ml.feature import Imputer
        from pyspark.sql import functions as F

        _require_columns(train.columns, ("building_zone", *MISSING_COLUMNS))
        if train.limit(1).count() == 0:
            raise ValueError("Training data is empty")
        if train.filter(F.col("building_zone").isNull()).limit(1).count():
            raise ValueError("Training building_zone contains null values")
        numeric = train
        for column in MISSING_COLUMNS:
            value = F.col(column).cast("double")
            numeric = numeric.withColumn(
                column,
                F.when(F.isnan(value) | (F.abs(value) == float("inf")), None)
                .otherwise(value),
            )
        numeric = numeric.cache()
        try:
            global_model = Imputer(
                inputCols=list(MISSING_COLUMNS),
                outputCols=[f"_global_{c}" for c in MISSING_COLUMNS],
                strategy="median",
            ).fit(numeric)
            self.global_medians_ = _surrogates(global_model, MISSING_COLUMNS)
            counts = numeric.groupBy("building_zone").agg(*[
                F.count(F.when(F.col(c).isNotNull() & ~F.isnan(F.col(c)), 1)).alias(c)
                for c in MISSING_COLUMNS
            ]).collect()
            self.zone_medians_ = {}
            for row in counts:
                zone = row["building_zone"]
                available = [c for c in MISSING_COLUMNS if row[c] > 0]
                medians = dict(self.global_medians_)
                if available:
                    model = Imputer(
                        inputCols=available,
                        outputCols=[f"_zone_{c}" for c in available],
                        strategy="median",
                    ).fit(numeric.filter(F.col("building_zone") == zone))
                    medians.update(_surrogates(model, available))
                self.zone_medians_[zone] = medians
        finally:
            numeric.unpersist()
        return self

    def transform(self, frame):
        from pyspark.sql import functions as F
        from pyspark.sql.types import DoubleType, StringType, StructField, StructType

        if not hasattr(self, "zone_medians_"):
            raise RuntimeError("Call fit(train) before transform(frame)")
        _require_columns(frame.columns, ("building_zone", *MISSING_COLUMNS,
                                         *PHYSICAL_INPUT_COLUMNS))
        if frame.filter(F.col("building_zone").isNull()).limit(1).count():
            raise ValueError("building_zone contains null values")
        spark = frame.sparkSession
        lookup_schema = StructType([
            StructField("_zone_key", StringType(), False),
            *(StructField(f"_median_{c}", DoubleType(), False) for c in MISSING_COLUMNS),
        ])
        rows = [tuple([str(zone)] + [float(values[c]) for c in MISSING_COLUMNS])
                for zone, values in self.zone_medians_.items()]
        lookup = spark.createDataFrame(rows, lookup_schema)
        result = frame.join(
            F.broadcast(lookup),
            frame["building_zone"] == lookup["_zone_key"],
            "left",
        ).drop("_zone_key")
        for column in (*MISSING_COLUMNS, *PHYSICAL_INPUT_COLUMNS):
            value = F.col(column).cast("double")
            result = result.withColumn(
                column,
                F.when(F.isnan(value) | (F.abs(value) == float("inf")), None)
                .otherwise(value),
            )
        for column in MISSING_COLUMNS:
            result = result.withColumn(
                column,
                F.coalesce(F.col(column), F.col(f"_median_{column}"),
                           F.lit(float(self.global_medians_[column]))),
            ).drop(f"_median_{column}")
        result = result.withColumn(
            "delta_temp_c", F.col("indoor_temp_c") - F.col("outdoor_temp_c")
        ).withColumn(
            "delta_humidity_pct",
            F.col("indoor_humidity_pct") - F.col("outdoor_humidity_pct"),
        )
        for output, numerator, denominator in (
            ("pm25_indoor_outdoor_ratio", "pm25_ug_m3", "outdoor_pm25_ug_m3"),
            ("filter_pressure_per_airflow", "filter_pressure_pa", "air_flow_m3_h"),
            ("hvac_kw_per_airflow", "hvac_power_kw", "air_flow_m3_h"),
        ):
            result = result.withColumn(
                output,
                F.when(F.col(denominator) > 0,
                       F.col(numerator) / F.col(denominator)).otherwise(None),
            )
        return result

    def fit_transform(self, train):
        return self.fit(train).transform(train)


def _surrogates(model, columns):
    values = model.surrogateDF.first().asDict()
    return {column: float(values[column]) for column in columns}


def correlation_pandas(train):
    """Pairwise Pearson correlations, with one binary column per target class."""
    import pandas as pd

    _require_columns(train.columns, (*ENGINEERED_COLUMNS, "diagnostic_state"))
    columns = {c: pd.to_numeric(train[c], errors="coerce") for c in ENGINEERED_COLUMNS}
    for label in LABELS:
        columns[f"state_{label}"] = train["diagnostic_state"].eq(label).astype(float)
    return pd.DataFrame(columns).corr().fillna(0.0)


def correlation_spark(train):
    """Pairwise Pearson correlations as one Spark aggregate over training rows."""
    import pandas as pd
    from pyspark.sql import functions as F

    _require_columns(train.columns, (*ENGINEERED_COLUMNS, "diagnostic_state"))
    df = train
    names = list(ENGINEERED_COLUMNS)
    for label in LABELS:
        name = f"state_{label}"
        names.append(name)
        df = df.withColumn(name, F.when(F.col("diagnostic_state") == label, 1.0).otherwise(0.0))
    expressions = [F.corr(F.col(a), F.col(b)).alias(f"c{i}_{j}")
                   for i, a in enumerate(names) for j, b in enumerate(names) if j >= i]
    row = df.agg(*expressions).first()
    matrix = pd.DataFrame(1.0, index=names, columns=names)
    for i, a in enumerate(names):
        for j in range(i, len(names)):
            b = names[j]
            value = row[f"c{i}_{j}"]
            matrix.loc[a, b] = matrix.loc[b, a] = float(value) if value is not None else 0.0
    return matrix


def write_analysis(matrix, output_dir):
    """Export reproducible training-only associations and an SVG heatmap."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_dir / "feature_correlation_matrix.csv", float_format="%.5f")
    labels = [f"state_{label}" for label in LABELS]
    association = matrix.loc[list(ENGINEERED_COLUMNS), labels].copy()
    association["max_abs_class_correlation"] = association.abs().max(axis=1)
    association = association.sort_values("max_abs_class_correlation", ascending=False)
    association.to_csv(output_dir / "feature_class_association.csv", float_format="%.5f")
    _write_svg_heatmap(matrix, output_dir / "feature_correlation_matrix.svg")
    return association


def _write_svg_heatmap(matrix, path):
    names = list(matrix.columns)
    size, left, top = 64, 310, 250
    width, height = left + size * len(names) + 160, top + size * len(names) + 55
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#152238}.axis{font-size:13px}.val{font-size:12px;font-weight:bold}.title{font-size:20px;font-weight:bold}</style>',
             '<text class="title" x="24" y="36">Training feature correlation matrix</text>',
             '<text x="24" y="61" font-size="12">Pearson r; class columns are one-vs-rest indicators</text>']
    for i, name in enumerate(names):
        y = top + i * size + size / 2 + 5
        parts.append(f'<text class="axis" x="{left - 10}" y="{y}" text-anchor="end">{escape(name)}</text>')
        x = left + i * size + size / 2
        parts.append(f'<text class="axis" transform="translate({x},{top - 10}) rotate(-55)" text-anchor="start">{escape(name)}</text>')
    for i, row_name in enumerate(names):
        for j, col_name in enumerate(names):
            value = float(matrix.loc[row_name, col_name])
            if not math.isfinite(value):
                value = 0.0
            strength = min(1.0, abs(value))
            color = (int(244 - 171 * strength), int(247 - 91 * strength),
                     int(250 - 31 * strength)) if value >= 0 else (
                     int(247 - 26 * strength), int(246 - 115 * strength),
                     int(244 - 174 * strength))
            fill = "#%02x%02x%02x" % color
            x, y = left + j * size, top + i * size
            parts.append(f'<rect x="{x}" y="{y}" width="{size}" height="{size}" fill="{fill}" stroke="#d5dce5"/>')
            parts.append(f'<text class="val" x="{x + size / 2}" y="{y + size / 2 + 4}" text-anchor="middle">{value:+.2f}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("pandas", "spark"), default="spark")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Raw CSV; split by recorded_at")
    source.add_argument("--train", help="Already split training CSV or Parquet")
    parser.add_argument("--test", help="Test CSV or Parquet, required with --train")
    parser.add_argument("--cutoff", default="2025-10-01",
                        help="Test starts at this ISO date (default: 2025-10-01)")
    parser.add_argument("--output-dir", default="data/processed/tv2")
    parser.add_argument("--charts-dir", default="docs/charts/tv2")
    parser.add_argument("--spark-master", default="local[*]")
    return parser.parse_args()


def _validate_cutoff(cutoff):
    from datetime import date
    date.fromisoformat(cutoff)


def _read_pandas(path):
    import pandas as pd

    return pd.read_parquet(path) if str(path).lower().endswith(".parquet") else pd.read_csv(path)


def _run_pandas(args):
    import pandas as pd

    if args.input:
        data = _read_pandas(args.input)
        _require_columns(data.columns, ("recorded_at", "diagnostic_state"))
        when = pd.to_datetime(data["recorded_at"], errors="raise")
        train, test = data.loc[when < args.cutoff], data.loc[when >= args.cutoff]
    else:
        train, test = _read_pandas(args.train), _read_pandas(args.test)
    if train.empty or test.empty:
        raise ValueError("Train or test split is empty; inspect --cutoff and input dates")
    engineer = PandasZoneMedianFeatureEngineer().fit(train)
    train_out, test_out = engineer.transform(train), engineer.transform(test)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_out.to_csv(output / "train_features.csv", index=False)
    test_out.to_csv(output / "test_features.csv", index=False)
    matrix = correlation_pandas(train_out)
    association = write_analysis(matrix, args.charts_dir)
    print(f"Train: {len(train_out):,}; test: {len(test_out):,}")
    print(association.to_string(float_format=lambda v: f"{v:.3f}"))


def _run_spark(args):
    from pyspark.sql import SparkSession, functions as F

    spark = SparkSession.builder.master(args.spark_master).appName(
        "IoTBuildingFeatureEngineering"
    ).getOrCreate()
    try:
        def read(path):
            return spark.read.parquet(path) if str(path).lower().endswith(".parquet") else (
                spark.read.option("header", True).option("inferSchema", True).csv(path)
            )

        if args.input:
            data = read(args.input)
            _require_columns(data.columns, ("recorded_at", "diagnostic_state"))
            data = data.withColumn("_event_time", F.to_timestamp("recorded_at"))
            if data.filter(F.col("_event_time").isNull()).limit(1).count():
                raise ValueError("recorded_at contains invalid timestamps")
            cutoff = F.to_timestamp(F.lit(args.cutoff))
            train = data.filter(F.col("_event_time") < cutoff).drop("_event_time")
            test = data.filter(F.col("_event_time") >= cutoff).drop("_event_time")
        else:
            train, test = read(args.train), read(args.test)
        if train.limit(1).count() == 0 or test.limit(1).count() == 0:
            raise ValueError("Train or test split is empty; inspect --cutoff and input dates")
        engineer = SparkZoneMedianFeatureEngineer().fit(train)
        train_out, test_out = engineer.transform(train), engineer.transform(test)
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        train_out.write.mode("overwrite").parquet(str(output / "train_features.parquet"))
        test_out.write.mode("overwrite").parquet(str(output / "test_features.parquet"))
        matrix = correlation_spark(train_out)
        association = write_analysis(matrix, args.charts_dir)
        print(f"Train: {train_out.count():,}; test: {test_out.count():,}")
        print(association.to_string(float_format=lambda v: f"{v:.3f}"))
    finally:
        spark.stop()


def main():
    args = _args()
    if args.train and not args.test:
        raise SystemExit("--test is required when --train is used")
    _validate_cutoff(args.cutoff)
    (_run_spark if args.engine == "spark" else _run_pandas)(args)
