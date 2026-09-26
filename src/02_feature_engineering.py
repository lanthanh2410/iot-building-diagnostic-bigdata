"""Command line entry point for member 2 feature engineering.

Examples:
    python src/02_feature_engineering.py --engine spark --input data/raw/11_iot_building_diagnostic.csv
    python src/02_feature_engineering.py --engine pandas --input data/raw/11_iot_building_diagnostic.csv
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import main


if __name__ == "__main__":
    main()
