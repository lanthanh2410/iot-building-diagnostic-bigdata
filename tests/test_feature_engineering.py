"""Behavior checks for the member 2 preprocessing contract."""

import unittest

import pandas as pd

from src.feature_engineering import (
    PandasZoneMedianFeatureEngineer,
    correlation_pandas,
    ENGINEERED_COLUMNS,
    MISSING_COLUMNS,
)


def row(zone, occupancy, *, outside_pm=10, airflow=100):
    return {
        "building_zone": zone,
        "occupancy_count": occupancy,
        "tvoc_ppb": occupancy,
        "vibration_mm_s": occupancy,
        "window_open_pct": occupancy,
        "indoor_temp_c": 25,
        "outdoor_temp_c": 20,
        "indoor_humidity_pct": 50,
        "outdoor_humidity_pct": 60,
        "pm25_ug_m3": 20,
        "outdoor_pm25_ug_m3": outside_pm,
        "filter_pressure_pa": 200,
        "air_flow_m3_h": airflow,
        "hvac_power_kw": 5,
        "diagnostic_state": "normal",
    }


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame([
            row("A", 1), row("A", 3), row("A", None),
            row("B", 10), row("B", 20), row("B", None),
        ])

    def test_zone_medians_training_only_and_unseen_fallback(self):
        engineer = PandasZoneMedianFeatureEngineer().fit(self.train)
        future = pd.DataFrame([
            row("A", None), row("B", None), row("C", None),
        ])
        future.loc[:, "diagnostic_state"] = "thermal_issue"
        transformed = engineer.transform(future)
        self.assertEqual(transformed["occupancy_count"].tolist(), [2, 15, 6.5])
        self.assertEqual(transformed["diagnostic_state"].tolist(),
                         ["thermal_issue"] * 3)
        self.assertEqual(ENGINEERED_COLUMNS[0], "delta_temp_c")
        self.assertEqual(transformed["delta_temp_c"].tolist(), [5, 5, 5])
        self.assertEqual(transformed["delta_humidity_pct"].tolist(), [-10, -10, -10])
        self.assertEqual(transformed["pm25_indoor_outdoor_ratio"].tolist(), [2, 2, 2])
        self.assertEqual(transformed["filter_pressure_per_airflow"].tolist(), [2, 2, 2])
        self.assertEqual(transformed["hvac_kw_per_airflow"].tolist(), [0.05] * 3)

    def test_zero_denominators_become_missing(self):
        engineer = PandasZoneMedianFeatureEngineer().fit(self.train)
        transformed = engineer.transform(pd.DataFrame([row("A", None, outside_pm=0,
                                                          airflow=0)]))
        self.assertTrue(pd.isna(transformed.loc[0, "pm25_indoor_outdoor_ratio"]))
        self.assertTrue(pd.isna(transformed.loc[0, "filter_pressure_per_airflow"]))
        self.assertTrue(pd.isna(transformed.loc[0, "hvac_kw_per_airflow"]))
        self.assertTrue(transformed[list(MISSING_COLUMNS)].notna().all().all())

    def test_all_missing_zone_column_uses_global_training_median(self):
        train = self.train.copy()
        train.loc[train["building_zone"] == "B", "tvoc_ppb"] = None
        engineer = PandasZoneMedianFeatureEngineer().fit(train)
        future = pd.DataFrame([row("B", None)])
        self.assertEqual(engineer.transform(future).loc[0, "tvoc_ppb"], 2)

    def test_correlation_has_one_indicator_per_class(self):
        train = self.train.copy()
        train.loc[3:, "diagnostic_state"] = "ventilation_issue"
        transformed = PandasZoneMedianFeatureEngineer().fit_transform(train)
        matrix = correlation_pandas(transformed)
        self.assertEqual(matrix.shape, (8, 8))
        self.assertIn("state_thermal_issue", matrix.columns)
        self.assertTrue((matrix.values == matrix.values.T).all())


if __name__ == "__main__":
    unittest.main()
