"""
Thành viên 4: Kiến trúc Hạ tầng Phân tán & Spark Feature Pipeline.
================================================================================
Đề tài 11: IoT Building Diagnostics (Chẩn đoán Vận hành Tòa nhà Thông minh)
Quy mô dữ liệu: 2.100.000 bản ghi, 24 thuộc tính.

Nhiệm vụ trọng tâm:
1. Cài đặt và cấu hình môi trường tính toán phân tán PySpark (Local Multi-core & Docker Spark Cluster).
2. Xây dựng Spark Data Pipeline phân tán:
   - StringIndexer: Mã hóa nhãn mục tiêu (diagnostic_state -> label) và vị trí (building_zone -> building_zone_idx).
   - Imputer: Điền khuyết các cột thiếu bằng median phân tán.
   - VectorAssembler: Gom toàn bộ 23 đặc trưng thành vector raw_features.
   - StandardScaler: Chuẩn hóa Z-score phân tán trên RDD/DataFrame thành vector features.
3. Tối ưu hóa hiệu năng I/O Big Data trên 2.1 triệu dòng:
   - Cấu hình số phân vùng (numPartitions) thích ứng với số CPU cores.
   - Cơ chế bộ nhớ đệm Caching: persist(StorageLevel.MEMORY_AND_DISK).
   - Tối ưu hóa cơ chế Shuffle trong Spark SQL và Spark ML (AQE, Kryo, dynamic partitions).
4. Thực nghiệm Benchmark và xuất Log chi tiết phân tích tác động của Partitioning và Caching.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Cấu hình encoding utf-8 cho console Windows để tránh UnicodeEncodeError
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Đảm bảo import được các module từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Cấu hình HADOOP_HOME tự động cho hệ điều hành Windows để hỗ trợ I/O Spark
if os.name == "nt" and "HADOOP_HOME" not in os.environ:
    tools_hadoop = ROOT_DIR / "tools" / "hadoop"
    if (tools_hadoop / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(tools_hadoop)
        os.environ["PATH"] = str(tools_hadoop / "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType, StructField, StructType, TimestampType
)
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import (
    StringIndexer,
    StringIndexerModel,
    VectorAssembler,
    StandardScaler,
    StandardScalerModel,
    Imputer,
    ImputerModel,
)

from src.utils.logger import ResourceLogger
from src.utils.spark_utils import (
    SparkSessionManager,
    cache_dataframe,
    unpersist_dataframe,
    get_optimal_partitions_count,
)

# ==============================================================================
# HẰNG SỐ CẤU TRÚC DỮ LIỆU & ĐẶC TẢ THUỘC TÍNH (Từ Điển Dữ Liệu Đề Tài 11)
# ==============================================================================

TARGET_COLUMN = "diagnostic_state"
LABEL_INDEX_COL = "label"
ZONE_COLUMN = "building_zone"
ZONE_INDEX_COL = "building_zone_idx"

# 3 nhãn phân loại đa lớp
LABELS = ["normal", "ventilation_issue", "thermal_issue"]

# 4 cột cảm biến có giá trị khuyết cần điền median
MISSING_COLUMNS = [
    "occupancy_count",
    "tvoc_ppb",
    "vibration_mm_s",
    "window_open_pct",
]

# Các cột cảm biến cơ bản liên quan nhiệt & thông gió
PHYSICAL_RAW_COLUMNS = [
    "indoor_temp_c",
    "outdoor_temp_c",
    "indoor_humidity_pct",
    "outdoor_humidity_pct",
    "pm25_ug_m3",
    "outdoor_pm25_ug_m3",
    "filter_pressure_pa",
    "air_flow_m3_h",
    "hvac_power_kw",
]

# Các cột thông số vận hành khác
EQUIPMENT_RAW_COLUMNS = [
    "floor_number",
    "co2_ppm",
    "fan_speed_rpm",
    "maintenance_days",
    "equipment_age_years",
]

# 5 đặc trưng vật lý IoT nâng cao (Thành viên 2)
ENGINEERED_COLUMNS = [
    "delta_temp_c",
    "delta_humidity_pct",
    "pm25_indoor_outdoor_ratio",
    "filter_pressure_per_airflow",
    "hvac_kw_per_airflow",
]

# Toàn bộ danh sách các đặc trưng số sẽ được gom vào VectorAssembler
# (Bao gồm đặc trưng thô, đặc trưng phái sinh, và chỉ mục khu vực building_zone_idx)
NUMERIC_FEATURE_COLUMNS = (
    ["floor_number"]
    + MISSING_COLUMNS
    + PHYSICAL_RAW_COLUMNS
    + ["co2_ppm", "fan_speed_rpm", "maintenance_days", "equipment_age_years"]
    + ENGINEERED_COLUMNS
)

# Định nghĩa Schema tường minh cho dữ liệu raw 2.1 triệu dòng
# (Định nghĩa Schema giúp Spark đọc nhanh hơn 3-5 lần so với inferSchema)
RAW_DATA_SCHEMA = StructType([
    StructField("reading_id", IntegerType(), True),
    StructField("recorded_at", StringType(), True),
    StructField("building_zone", StringType(), True),
    StructField("sensor_id", StringType(), True),
    StructField("floor_number", IntegerType(), True),
    StructField("occupancy_count", DoubleType(), True),
    StructField("outdoor_temp_c", DoubleType(), True),
    StructField("outdoor_humidity_pct", DoubleType(), True),
    StructField("outdoor_pm25_ug_m3", DoubleType(), True),
    StructField("indoor_temp_c", DoubleType(), True),
    StructField("indoor_humidity_pct", DoubleType(), True),
    StructField("co2_ppm", DoubleType(), True),
    StructField("pm25_ug_m3", DoubleType(), True),
    StructField("tvoc_ppb", DoubleType(), True),
    StructField("air_flow_m3_h", DoubleType(), True),
    StructField("fan_speed_rpm", DoubleType(), True),
    StructField("hvac_power_kw", DoubleType(), True),
    StructField("filter_pressure_pa", DoubleType(), True),
    StructField("vibration_mm_s", DoubleType(), True),
    StructField("maintenance_days", IntegerType(), True),
    StructField("window_open_pct", DoubleType(), True),
    StructField("equipment_age_years", DoubleType(), True),
    StructField("firmware_version", StringType(), True),
    StructField("diagnostic_state", StringType(), True),
])


# ==============================================================================
# 1. TIỀN XỬ LÝ & KỸ NGHỆ ĐẶC TRƯNG VẬT LÝ TRÊN SPARK DATAFRAME
# ==============================================================================

def add_engineered_physical_features(df: DataFrame) -> DataFrame:
    """
    Tính toán 5 đặc trưng vật lý IoT phân tán bằng biểu thức Spark SQL (nếu chưa có).
    Xử lý an toàn mẫu số <= 0 để tránh vô cực (Inf/NaN).
    """
    existing_cols = set(df.columns)
    result_df = df

    # Ép kiểu dữ liệu số an toàn
    for col_name in MISSING_COLUMNS + PHYSICAL_RAW_COLUMNS:
        if col_name in existing_cols:
            val_col = F.col(col_name).cast("double")
            result_df = result_df.withColumn(
                col_name,
                F.when(F.isnan(val_col) | (F.abs(val_col) == float("inf")), None).otherwise(val_col)
            )

    # 1. Chênh lệch nhiệt độ trong/ngoài: delta_temp_c = indoor_temp_c - outdoor_temp_c
    if "delta_temp_c" not in existing_cols and "indoor_temp_c" in existing_cols and "outdoor_temp_c" in existing_cols:
        result_df = result_df.withColumn("delta_temp_c", F.col("indoor_temp_c") - F.col("outdoor_temp_c"))

    # 2. Chênh lệch độ ẩm trong/ngoài: delta_humidity_pct = indoor_humidity_pct - outdoor_humidity_pct
    if "delta_humidity_pct" not in existing_cols and "indoor_humidity_pct" in existing_cols and "outdoor_humidity_pct" in existing_cols:
        result_df = result_df.withColumn("delta_humidity_pct", F.col("indoor_humidity_pct") - F.col("outdoor_humidity_pct"))

    # 3. Tỷ lệ bụi mịn trong/ngoài: pm25_indoor_outdoor_ratio = pm25_ug_m3 / outdoor_pm25_ug_m3
    if "pm25_indoor_outdoor_ratio" not in existing_cols and "pm25_ug_m3" in existing_cols and "outdoor_pm25_ug_m3" in existing_cols:
        result_df = result_df.withColumn(
            "pm25_indoor_outdoor_ratio",
            F.when(F.col("outdoor_pm25_ug_m3") > 0, F.col("pm25_ug_m3") / F.col("outdoor_pm25_ug_m3")).otherwise(None)
        )

    # 4. Trở lực lọc trên lưu lượng gió: filter_pressure_per_airflow = filter_pressure_pa / air_flow_m3_h
    if "filter_pressure_per_airflow" not in existing_cols and "filter_pressure_pa" in existing_cols and "air_flow_m3_h" in existing_cols:
        result_df = result_df.withColumn(
            "filter_pressure_per_airflow",
            F.when(F.col("air_flow_m3_h") > 0, F.col("filter_pressure_pa") / F.col("air_flow_m3_h")).otherwise(None)
        )

    # 5. Suất năng lượng HVAC: hvac_kw_per_airflow = hvac_power_kw / air_flow_m3_h
    if "hvac_kw_per_airflow" not in existing_cols and "hvac_power_kw" in existing_cols and "air_flow_m3_h" in existing_cols:
        result_df = result_df.withColumn(
            "hvac_kw_per_airflow",
            F.when(F.col("air_flow_m3_h") > 0, F.col("hvac_power_kw") / F.col("air_flow_m3_h")).otherwise(None)
        )

    return result_df


# ==============================================================================
# 2. XÂY DỰNG SPARK FEATURE PIPELINE PHÂN TÁN
# ==============================================================================

def build_spark_feature_pipeline(
    df_columns: List[str],
    include_label: bool = True,
    with_std: bool = True,
    with_mean: bool = False,
) -> Tuple[Pipeline, List[str]]:
    """
    Xây dựng Spark ML Pipeline gồm 4 công đoạn chuẩn:
    1. StringIndexer:
       - building_zone -> building_zone_idx (24 khu vực)
       - diagnostic_state -> label (nếu include_label=True)
    2. Imputer:
       - Điền giá trị khuyết cho 4 cột cảm biến bằng giá trị median toàn cục.
    3. VectorAssembler:
       - Gom toàn bộ đặc trưng số + building_zone_idx thành vector raw_features.
    4. StandardScaler:
       - Chuẩn hóa Z-score phân tán: raw_features -> features.

    Returns:
    --------
    pipeline : pyspark.ml.Pipeline
    assembled_feature_names : List[str] danh sách các cột thành phần của vector đặc trưng.
    """
    stages = []

    # 1. StringIndexer cho vị trí khu vực (building_zone)
    zone_indexer = StringIndexer(
        inputCol=ZONE_COLUMN,
        outputCol=ZONE_INDEX_COL,
        handleInvalid="keep",  # Giữ lại nhãn mới trong tương lai thành index đặc biệt thay vì crash
        stringOrderType="alphabetAsc",
    )
    stages.append(zone_indexer)

    # 2. StringIndexer cho nhãn mục tiêu (diagnostic_state)
    if include_label and TARGET_COLUMN in df_columns:
        label_indexer = StringIndexer(
            inputCol=TARGET_COLUMN,
            outputCol=LABEL_INDEX_COL,
            handleInvalid="keep",
            stringOrderType="frequencyDesc",
        )
        stages.append(label_indexer)

    # 3. Imputer: Điền khuyết phân tán bằng median cho các cột có null
    cols_to_impute = [col for col in MISSING_COLUMNS if col in df_columns]
    if cols_to_impute:
        imputer = Imputer(
            inputCols=cols_to_impute,
            outputCols=cols_to_impute,  # Điền trực tiếp vào cột hiện tại
            strategy="median",
        )
        stages.append(imputer)

    # 4. Xác định danh sách đặc trưng đầu vào cho VectorAssembler
    # Lấy các cột số thực tế tồn tại trong DataFrame + building_zone_idx
    selected_numeric_cols = [c for c in NUMERIC_FEATURE_COLUMNS if c in df_columns]
    assembled_feature_names = selected_numeric_cols + [ZONE_INDEX_COL]

    # VectorAssembler: Gom các cột thành một Dense/Sparse Vector duy nhất
    vector_assembler = VectorAssembler(
        inputCols=assembled_feature_names,
        outputCol="raw_features",
        handleInvalid="keep",  # Không loại bỏ dòng, giữ các giá trị nan dưới dạng vector giá trị
    )
    stages.append(vector_assembler)

    # 5. StandardScaler: Chuẩn hóa phân tán trên DataFrame
    # withStd=True: chia cho độ lệch chuẩn.
    # withMean=False: tránh chuyển SparseVector thành DenseVector trong bộ nhớ để tiết kiệm RAM.
    standard_scaler = StandardScaler(
        inputCol="raw_features",
        outputCol="features",
        withStd=with_std,
        withMean=with_mean,
    )
    stages.append(standard_scaler)

    pipeline = Pipeline(stages=stages)
    return pipeline, assembled_feature_names


# ==============================================================================
# 3. TỐI ƯU HÓA HIỆU NĂNG I/O BIG DATA & PARTITIONING
# ==============================================================================

def optimize_dataframe_partitions(
    df: DataFrame,
    spark: SparkSession,
    target_partitions: Optional[int] = None,
    log_changes: bool = True,
) -> DataFrame:
    """
    Tối ưu hóa số lượng phân vùng (partitions) của DataFrame phù hợp với CPU cores.
    Nếu số partition quá lớn (ví dụ sau shuffle bị tạo 200 partition nhỏ) -> Coalesce.
    Nếu số partition quá nhỏ (ví dụ đọc từ 1 file CSV lớn) -> Repartition để chạy song song đa luồng.
    """
    current_partitions = df.rdd.getNumPartitions()
    num_cores = spark.sparkContext.defaultParallelism

    if target_partitions is None:
        target_partitions = max(4, num_cores)

    if current_partitions == target_partitions:
        return df

    if current_partitions > target_partitions:
        # Giảm phân vùng bằng coalesce (không gây full shuffle, nhanh và ít tốn RAM)
        optimized_df = df.coalesce(target_partitions)
        action = f"COALESCE ({current_partitions} -> {target_partitions})"
    else:
        # Tăng phân vùng bằng repartition (phân tán đều dữ liệu sang các worker/core)
        optimized_df = df.repartition(target_partitions)
        action = f"REPARTITION ({current_partitions} -> {target_partitions})"

    if log_changes:
        print(f" [Big Data I/O Optimizer] {action} dựa trên {num_cores} cores.")

    return optimized_df


# ==============================================================================
# 4. BỘ ĐO LƯỜNG & PHÂN TÍCH THỰC NGHIỆM: PARTITIONING & CACHING BENCHMARK
# ==============================================================================

class PipelineBenchmarkRunner:
    """
    Thực nghiệm phân tích tác động của Partitioning và Caching lên thời gian xử lý
    trên tập dữ liệu quy mô lớn (2.1 triệu dòng hoặc mẫu kiểm thử).
    """

    def __init__(self, spark: SparkSession, log_file_path: Path):
        self.spark = spark
        self.log_file_path = log_file_path
        self.logger = ResourceLogger()
        self.log_file_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, text: str) -> None:
        """In ra console đồng thời ghi vào file log."""
        print(text)
        with open(self.log_file_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def run_benchmark(self, raw_df: DataFrame) -> Dict[str, Any]:
        """Thực hiện toàn bộ kịch bản kiểm thử đo lường hiệu năng."""
        num_records = raw_df.count()
        num_cores = self.spark.sparkContext.defaultParallelism

        border = "=" * 80
        self.log(border)
        self.log(f" BÁO CÁO THỰC NGHIỆM TỐI ƯU HÓA HIỆU NĂNG BIG DATA SPARK PIPELINE ")
        self.log(f" Thời gian thực hiện : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log(f" Quy mô dữ liệu test : {num_records:,} bản ghi | {len(raw_df.columns)} cột")
        self.log(f" Cấu hình phần cứng  : {num_cores} Logical Cores | Spark {self.spark.version}")
        self.log(border + "\n")

        results = {}

        # ----------------------------------------------------------------------
        # THỰC NGHIỆM 1: TÁC ĐỘNG CỦA CACHING (MEMORY_AND_DISK vs NO CACHE)
        # ----------------------------------------------------------------------
        self.log(">>> [THỰC NGHIỆM 1] Đo lường tác động của Caching (MEMORY_AND_DISK vs No Cache)...")
        self.log("Kịch bản: Thực hiện 3 tác vụ tính toán thống kê (Aggregation + Pipeline Transform) lặp lại.\n")

        # Kịch bản 1.1: KHÔNG DÙNG CACHE (No Cache)
        uncached_df = raw_df.persist(StorageLevel.NONE)
        pipeline, _ = build_spark_feature_pipeline(uncached_df.columns, include_label=True)

        self.logger.start()
        t0 = time.time()
        # Vòng 1: Fit pipeline
        model_uncached = pipeline.fit(uncached_df)
        trans_uncached = model_uncached.transform(uncached_df)
        # Vòng 2: Action đếm & tổng hợp phân bố nhãn
        agg_uncached_1 = trans_uncached.groupBy("label").count().collect()
        # Vòng 3: Action tính giá trị trung bình trên 1 đặc trưng
        agg_uncached_2 = trans_uncached.select(F.avg("indoor_temp_c")).collect()
        t_no_cache = time.time() - t0
        metrics_no_cache = self.logger.stop()

        self.log(f" • [Kịch bản A - NO CACHE]:")
        self.log(f"   - Tổng thời gian thực thi : {t_no_cache:.4f} giây")
        self.log(f"   - Tốc độ xử lý (Throughput): {num_records / t_no_cache:,.0f} records/giây")
        self.log(f"   - RAM sử dụng thêm        : {metrics_no_cache['ram_used_mb']:.2f} MB\n")

        # Kịch bản 1.2: CÓ DÙNG CACHE (StorageLevel.MEMORY_AND_DISK)
        cached_df = raw_df.persist(StorageLevel.MEMORY_AND_DISK)
        cached_df.count()  # Materialize DataFrame vào cache

        self.logger.start()
        t0 = time.time()
        # Vòng 1: Fit pipeline trên cached data
        model_cached = pipeline.fit(cached_df)
        trans_cached = model_cached.transform(cached_df)
        # Vòng 2: Action đếm & tổng hợp phân bố nhãn (tận dụng cache)
        agg_cached_1 = trans_cached.groupBy("label").count().collect()
        # Vòng 3: Action tính giá trị trung bình trên 1 đặc trưng (tận dụng cache)
        agg_cached_2 = trans_cached.select(F.avg("indoor_temp_c")).collect()
        t_cache = time.time() - t0
        metrics_cache = self.logger.stop()

        speedup_caching = t_no_cache / t_cache if t_cache > 0 else 1.0

        self.log(f" • [Kịch bản B - WITH CACHE (MEMORY_AND_DISK)]:")
        self.log(f"   - Tổng thời gian thực thi : {t_cache:.4f} giây")
        self.log(f"   - Tốc độ xử lý (Throughput): {num_records / t_cache:,.0f} records/giây")
        self.log(f"   - RAM sử dụng thêm        : {metrics_cache['ram_used_mb']:.2f} MB")
        self.log(f"   -> HIỆU QUẢ CACHING       : Tăng tốc {speedup_caching:.2f}x lần ({((t_no_cache - t_cache) / t_no_cache) * 100:.1f}% thời gian tiết kiệm)\n")

        unpersist_dataframe(cached_df)

        # ----------------------------------------------------------------------
        # THỰC NGHIỆM 2: TÁC ĐỘNG CỦA PARTITIONING & SHUFFLE TUNING
        # ----------------------------------------------------------------------
        self.log(">>> [THỰC NGHIỆM 2] Đo lường tác động của Phân vùng (Partitioning) & Shuffle Tuning...")
        self.log("Kịch bản: Thực hiện GroupBy Aggregation & Join trên 2 mức phân vùng khác nhau.\n")

        # Phân vùng không tối ưu: 1 hoặc 2 partition (thiếu song song) hoặc 200 partition mặc định
        non_optimal_parts = 2
        df_low_part = raw_df.repartition(non_optimal_parts)

        t0 = time.time()
        agg_low = df_low_part.groupBy("building_zone").agg(
            F.avg("indoor_temp_c").alias("avg_temp"),
            F.avg("hvac_power_kw").alias("avg_power"),
            F.count("*").alias("count")
        ).collect()
        t_low_part = time.time() - t0

        # Phân vùng tối ưu: Bằng 2x - 3x số CPU Cores (ví dụ 12-24 partition)
        optimal_parts = max(4, num_cores * 2)
        df_optimal_part = raw_df.repartition(optimal_parts)

        t0 = time.time()
        agg_opt = df_optimal_part.groupBy("building_zone").agg(
            F.avg("indoor_temp_c").alias("avg_temp"),
            F.avg("hvac_power_kw").alias("avg_power"),
            F.count("*").alias("count")
        ).collect()
        t_opt_part = time.time() - t0

        speedup_partitioning = t_low_part / t_opt_part if t_opt_part > 0 else 1.0

        self.log(f" • [Kịch bản C - Non-optimal Partitions ({non_optimal_parts} partitions)]:")
        self.log(f"   - Thời gian GroupBy/Shuffle: {t_low_part:.4f} giây")
        self.log(f" • [Kịch bản D - Optimized Partitions ({optimal_parts} partitions = {num_cores} cores * 2)]:")
        self.log(f"   - Thời gian GroupBy/Shuffle: {t_opt_part:.4f} giây")
        self.log(f"   -> HIỆU QUẢ PHÂN VÙNG      : Tăng tốc {speedup_partitioning:.2f}x lần\n")

        # ----------------------------------------------------------------------
        # BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG
        # ----------------------------------------------------------------------
        summary_table = [
            "| Tiêu chí Thử nghiệm | Cấu hình Thử nghiệm | Thời gian (giây) | Throughput (dòng/s) | Tốc độ tương đối |",
            "| :--- | :--- | :---: | :---: | :---: |",
            f"| Caching Effect | No Cache (Đọc lại từ đầu) | {t_no_cache:.4f}s | {num_records / t_no_cache:,.0f} | 1.00x (Gốc) |",
            f"| Caching Effect | MEMORY_AND_DISK | {t_cache:.4f}s | {num_records / t_cache:,.0f} | **{speedup_caching:.2f}x nhanh hơn** |",
            f"| Partitioning Effect | 2 Partitions (Chưa tối ưu) | {t_low_part:.4f}s | {num_records / t_low_part:,.0f} | 1.00x (Gốc) |",
            f"| Partitioning Effect | {optimal_parts} Partitions (Tối ưu theo Cores) | {t_opt_part:.4f}s | {num_records / t_opt_part:,.0f} | **{speedup_partitioning:.2f}x nhanh hơn** |",
        ]

        self.log(">>> [BẢNG KẾT QUẢ ĐỐI ĐẦU HIỆU NĂNG]")
        for row in summary_table:
            self.log(row)
        self.log("\n" + border + "\n")

        results = {
            "num_records": num_records,
            "t_no_cache": round(t_no_cache, 4),
            "t_cache": round(t_cache, 4),
            "speedup_caching": round(speedup_caching, 2),
            "t_low_part": round(t_low_part, 4),
            "t_opt_part": round(t_opt_part, 4),
            "speedup_partitioning": round(speedup_partitioning, 2),
            "optimal_partitions": optimal_parts,
        }
        return results


# ==============================================================================
# 5. LỚP ĐIỀU KHIỂN CHÍNH: SPARK FEATURE PIPELINE ENGINE
# ==============================================================================

class SparkFeaturePipelineEngine:
    """
    Engine điều phối Feature Pipeline phân tán phục vụ huấn luyện mô hình MLlib.
    """

    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.pipeline: Optional[Pipeline] = None
        self.pipeline_model: Optional[PipelineModel] = None
        self.feature_names: List[str] = []
        self.labels: List[str] = LABELS

    def load_data(
        self,
        data_path: str,
        explicit_schema: bool = True,
        partitions: Optional[int] = None,
    ) -> DataFrame:
        """
        Đọc dữ liệu lớn (CSV hoặc Parquet) với hiệu năng tối ưu.
        Định nghĩa Schema tường minh cho CSV để loại bỏ overhead inferSchema trên 2.1 triệu dòng.
        """
        path_str = str(data_path)
        print(f"-> Đang tải dữ liệu từ: {path_str}")

        if path_str.endswith(".parquet") or os.path.isdir(path_str):
            df = self.spark.read.parquet(path_str)
        else:
            reader = self.spark.read.option("header", "true")
            if explicit_schema:
                # Dùng schema định nghĩa trước giúp Spark không phải duyệt qua toàn bộ file để đoán kiểu
                reader = reader.schema(RAW_DATA_SCHEMA)
            else:
                reader = reader.option("inferSchema", "true")
            df = reader.csv(path_str)

        # Tính toán bổ sung các đặc trưng vật lý nếu chưa có
        df = add_engineered_physical_features(df)

        # Tối ưu hóa phân vùng
        df = optimize_dataframe_partitions(df, self.spark, target_partitions=partitions)

        return df

    def fit_transform(
        self,
        train_df: DataFrame,
        test_df: Optional[DataFrame] = None,
        use_cache: bool = True,
    ) -> Tuple[DataFrame, Optional[DataFrame]]:
        """
        Fit Pipeline trên tập Train và Transform cho cả Train và Test.
        Đảm bảo không rò rỉ dữ liệu (Data Leakage) giữa Train và Test.
        """
        print("-> Đang xây dựng cấu trúc Spark Feature Pipeline...")
        self.pipeline, self.feature_names = build_spark_feature_pipeline(
            train_df.columns,
            include_label=True,
            with_std=True,
            with_mean=False,
        )

        if use_cache:
            train_df = cache_dataframe(train_df, StorageLevel.MEMORY_AND_DISK)
            train_df.count()

        print("-> Đang Fit Spark Pipeline trên tập Train...")
        start_time = time.time()
        self.pipeline_model = self.pipeline.fit(train_df)
        fit_duration = time.time() - start_time
        print(f" [Spark ML] Pipeline fit hoàn tất trong {fit_duration:.2f} giây.")

        print("-> Đang Transform tập Train...")
        transformed_train = self.pipeline_model.transform(train_df)

        transformed_test = None
        if test_df is not None:
            if use_cache:
                test_df = cache_dataframe(test_df, StorageLevel.MEMORY_AND_DISK)
                test_df.count()
            print("-> Đang Transform tập Test...")
            transformed_test = self.pipeline_model.transform(test_df)

        return transformed_train, transformed_test

    def save_pipeline_artifacts(self, output_dir: str) -> None:
        """
        Lưu mô hình PipelineModel và file metadata JSON phục vụ thành viên 5 và 6.
        """
        if self.pipeline_model is None:
            raise RuntimeError("Chưa huấn luyện PipelineModel. Hãy gọi fit_transform() trước!")

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model_save_path = str(out_path / "spark_feature_pipeline_model")

        print(f"-> Đang lưu Spark PipelineModel artifact vào: {model_save_path}")
        try:
            self.pipeline_model.write().overwrite().save(model_save_path)
            print(f" [Spark ML] Đã lưu PipelineModel thành công tại: {model_save_path}")
        except Exception as e:
            print(f" [Cảnh báo I/O] Không thể ghi PipelineModel nguyên khối ra đĩa ({e}).")
            print(" [Thông báo] Tiến trình vẫn sẽ xuất đầy đủ Feature Metadata JSON để phục vụ thành viên 5 và 6.")

        # Trích xuất metadata chỉ số đặc trưng (Feature Index Mapping)
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "target_column": TARGET_COLUMN,
            "label_index_column": LABEL_INDEX_COL,
            "labels": self.labels,
            "feature_vector_dimension": len(self.feature_names),
            "feature_names": self.feature_names,
            "missing_columns_imputed": MISSING_COLUMNS,
            "engineered_physical_columns": ENGINEERED_COLUMNS,
            "scaling": {
                "method": "StandardScaler",
                "with_std": True,
                "with_mean": False,
            }
        }

        metadata_file = out_path / "feature_metadata.json"
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        print(f" [Metadata] Đã xuất bản đồ đặc trưng ra: {metadata_file}")


# ==============================================================================
# 6. TIỆN ÍCH TẠO DỮ LIỆU TỔNG HỢP KIỂM THỬ TẢI LỚN (SYNTHETIC DATA GENERATOR)
# ==============================================================================

def generate_synthetic_spark_data(spark: SparkSession, num_rows: int = 50000) -> DataFrame:
    """
    Tạo nhanh Spark DataFrame tổng hợp đúng chuẩn 24 thuộc tính để kiểm thử pipeline
    và benchmark hiệu năng khi chưa có sẵn file 2.1 triệu dòng.
    """
    print(f"-> Đang tạo tập dữ liệu tổng hợp {num_rows:,} bản ghi chuẩn schema để kiểm thử...")
    import random
    from datetime import datetime, timedelta

    zones = [f"B{b:02d}-Z{z:02d}" for b in range(1, 7) for z in range(1, 5)]
    sensors = [f"S{s:04d}" for s in range(1, 121)]
    firmwares = ["v1.0.2", "v1.1.0", "v2.0.1", "v2.1.3"]
    labels = ["normal", "ventilation_issue", "thermal_issue"]
    label_weights = [0.75, 0.18, 0.07]

    base_time = datetime(2025, 1, 1, 0, 0)
    data = []

    for i in range(1, num_rows + 1):
        rec_time = (base_time + timedelta(minutes=30 * (i % 17520))).strftime("%Y-%m-%dT%H:%M")
        zone = random.choice(zones)
        sensor = random.choice(sensors)
        floor = int(zone.split("-")[1].replace("Z", ""))
        state = random.choices(labels, weights=label_weights)[0]

        out_temp = round(random.uniform(18.0, 36.0), 2)
        out_hum = round(random.uniform(45.0, 90.0), 2)
        out_pm25 = round(random.uniform(15.0, 85.0), 2)

        if state == "normal":
            in_temp = round(out_temp + random.uniform(-4.0, 2.0), 2)
            in_hum = round(random.uniform(45.0, 65.0), 2)
            co2 = round(random.uniform(420.0, 750.0), 2)
            pm25 = round(out_pm25 * random.uniform(0.3, 0.6), 2)
            airflow = round(random.uniform(1800.0, 3200.0), 2)
            hvac_kw = round(random.uniform(15.0, 40.0), 2)
            filter_pa = round(random.uniform(110.0, 220.0), 2)
        elif state == "ventilation_issue":
            in_temp = round(out_temp + random.uniform(-1.0, 5.0), 2)
            in_hum = round(random.uniform(60.0, 85.0), 2)
            co2 = round(random.uniform(1100.0, 2100.0), 2)
            pm25 = round(out_pm25 * random.uniform(0.9, 1.8), 2)
            airflow = round(random.uniform(600.0, 1400.0), 2)
            hvac_kw = round(random.uniform(35.0, 60.0), 2)
            filter_pa = round(random.uniform(320.0, 460.0), 2)
        else:
            in_temp = round(out_temp + random.uniform(5.5, 12.0), 2)
            in_hum = round(random.uniform(70.0, 92.0), 2)
            co2 = round(random.uniform(600.0, 1100.0), 2)
            pm25 = round(out_pm25 * random.uniform(0.4, 0.8), 2)
            airflow = round(random.uniform(1200.0, 2400.0), 2)
            hvac_kw = round(random.uniform(45.0, 70.0), 2)
            filter_pa = round(random.uniform(150.0, 280.0), 2)

        occ = None if random.random() < 0.04 else float(random.randint(0, 45))
        tvoc = None if random.random() < 0.04 else round(random.uniform(60.0, 450.0), 2)
        vib = None if random.random() < 0.04 else round(random.uniform(0.3, 3.8), 2)
        win_pct = None if random.random() < 0.04 else round(random.uniform(0.0, 80.0), 2)

        data.append((
            i, rec_time, zone, sensor, floor, occ, out_temp, out_hum, out_pm25,
            in_temp, in_hum, co2, pm25, tvoc, airflow, round(random.uniform(800.0, 1650.0), 2),
            hvac_kw, filter_pa, vib, random.randint(1, 150), win_pct,
            round(random.uniform(0.5, 9.5), 1), random.choice(firmwares), state
        ))

    return spark.createDataFrame(data, schema=RAW_DATA_SCHEMA)


# ==============================================================================
# 7. HÀM MAIN VÀ XỬ LÝ DÒNG LỆNH CLI
# ==============================================================================

def parse_arguments() -> argparse.Namespace:
    """Phân tích các tham số dòng lệnh CLI."""
    parser = argparse.ArgumentParser(
        description="Thành viên 4: Cấu hình Spark Session & Feature Pipeline phân tán."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Đường dẫn file dữ liệu đầu vào (.csv hoặc .parquet). Mặc định tự dò tìm.",
    )
    parser.add_argument(
        "--spark-master",
        type=str,
        default=None,
        help="Spark Master URL (VD: 'local[*]', 'local[4]', 'spark://localhost:7077').",
    )
    parser.add_argument(
        "--partitions",
        type=int,
        default=None,
        help="Số phân vùng numPartitions tùy chỉnh (mặc định tối ưu theo CPU cores).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/spark_feature_pipeline",
        help="Thư mục lưu artifact PipelineModel và metadata.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="docs/logs",
        help="Thư mục ghi log tối ưu hóa hiệu năng.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Kích hoạt bài đo lường hiệu năng chuyên sâu về Partitioning & Caching.",
    )
    parser.add_argument(
        "--synthetic-rows",
        type=int,
        default=0,
        help="Số bản ghi tổng hợp cần sinh để benchmark tải lớn (VD: 50000 hoặc 200000).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("\n" + "=" * 70)
    print(" KHỞI ĐỘNG MODULE SPARK FEATURE PIPELINE & CẤU HÌNH PHÂN TÁN ")
    print(" Thành viên 4: Kiến trúc Hạ tầng Phân tán & Tối ưu Big Data ")
    print("=" * 70 + "\n")

    # 1. Khởi tạo SparkSession tối ưu
    spark = SparkSessionManager.get_spark_session(
        app_name="IoT_Building_Diagnostic_SparkPipeline",
        master=args.spark_master,
        log_level="WARN",
    )
    SparkSessionManager.print_cluster_banner(spark)

    try:
        engine = SparkFeaturePipelineEngine(spark)

        # 2. Xác định nguồn dữ liệu
        data_df = None
        if args.synthetic_rows > 0:
            data_df = generate_synthetic_spark_data(spark, args.synthetic_rows)
        elif args.input and os.path.exists(args.input):
            data_df = engine.load_data(args.input, partitions=args.partitions)
        else:
            # Tự động tìm nguồn dữ liệu khả dụng theo thứ tự ưu tiên
            candidates = [
                "data/raw/11_iot_building_diagnostic.csv",
                "data/processed/tv2/train_features.parquet",
                "data/sample/sample_500_rows.csv",
            ]
            for cand in candidates:
                cand_path = Path(cand)
                if cand_path.exists():
                    print(f"-> Tìm thấy dữ liệu tại: {cand}")
                    data_df = engine.load_data(str(cand_path), partitions=args.partitions)
                    break

            if data_df is None:
                print(" [Thông báo] Không tìm thấy tệp dữ liệu có sẵn. Đang tự động tạo mẫu 10,000 dòng để chạy thử...")
                data_df = generate_synthetic_spark_data(spark, num_rows=10000)

        record_count = data_df.count()
        print(f" [Bộ dữ liệu] Tổng số bản ghi nạp thành công: {record_count:,}")
        print(f" [Bộ dữ liệu] Số phân vùng hiện tại: {data_df.rdd.getNumPartitions()} partitions")

        # 3. Chạy Benchmark phân tích tác động Partitioning và Caching (nếu được yêu cầu hoặc chạy mặc định)
        log_file = Path(args.log_dir) / "spark_pipeline_optimization.log"
        runner = PipelineBenchmarkRunner(spark, log_file)
        if args.benchmark or record_count >= 500:
            print("\n-> Bắt đầu đo lường thực nghiệm Partitioning & Caching...")
            runner.run_benchmark(data_df)

        # 4. Huấn luyện và Transform Feature Pipeline
        print("\n-> Thực thi Spark Feature Pipeline...")
        # Tách tập Train/Test theo tỷ lệ 80/20 hoặc theo thời gian nếu có
        train_df, test_df = data_df.randomSplit([0.8, 0.2], seed=42)
        transformed_train, transformed_test = engine.fit_transform(train_df, test_df, use_cache=True)

        print("\n-> Cấu trúc dữ liệu sau khi biến đổi qua Pipeline:")
        sample_row = transformed_train.select("building_zone", "building_zone_idx", "diagnostic_state", "label", "features").first()
        print(f"   • Khu vực           : {sample_row['building_zone']} -> Index: {sample_row['building_zone_idx']}")
        print(f"   • Trạng thái nhãn   : {sample_row['diagnostic_state']} -> Label: {sample_row['label']}")
        print(f"   • Feature Vector size: {sample_row['features'].size} dimensions")

        # 5. Lưu trữ Artifacts cho thành viên 5 & 6
        engine.save_pipeline_artifacts(args.output_dir)

        print("\n" + "=" * 70)
        print(" HOÀN TẤT THÀNH CÔNG MODULE 04_SPARK_PIPELINE.PY ")
        print(f" Log chi tiết đã lưu tại : {log_file}")
        print(f" Artifacts đã lưu tại    : {args.output_dir}/")
        print("=" * 70 + "\n")

    finally:
        spark.stop()
        print(" SparkSession đã đóng an toàn.")


if __name__ == "__main__":
    main()
