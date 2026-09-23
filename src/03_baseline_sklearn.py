"""
Thành viên 3: Xây dựng & Đánh giá Mô hình Baseline Đơn máy (Scikit-Learn).
Nhiệm vụ:
- Huấn luyện Logistic Regression / Random Forest / LightGBM.
- Đánh giá Accuracy, F1-score.
- Log tài nguyên RAM/CPU.
"""

import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier



def build_preprocessor(X):
    print("\n===== XÂY DỰNG PREPROCESSING PIPELINE =====")

    categorical_features = ["building_zone"]

    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                numeric_features
            ),
            (
                "categorical",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    print("Categorical features:")
    print(categorical_features)

    print("\nNumeric features:")
    print(numeric_features)

    return preprocessor

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RAW_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "11_iot_building_diagnostic.csv"
)

def test_preprocessor(preprocessor, X):
    print("\n===== TEST PREPROCESSING =====")

    # Chỉ dùng 5.000 dòng để test nhanh
    X_test_sample = X.head(5000)

    X_transformed = preprocessor.fit_transform(X_test_sample)

    print("Kích thước trước preprocessing:", X_test_sample.shape)
    print("Kích thước sau preprocessing:", X_transformed.shape)

    print("Preprocessing chạy thành công!")

    return X_transformed

def temporary_time_split(df, train_ratio=0.8):
    print("\n===== CHIA TRAIN / TEST TẠM THEO THỜI GIAN =====")

    df = df.sort_values("recorded_at").reset_index(drop=True)

    unique_times = df["recorded_at"].sort_values().unique()
    cutoff_index = int(len(unique_times) * train_ratio)
    cutoff_time = unique_times[cutoff_index]

    train_df = df[df["recorded_at"] < cutoff_time].copy()
    test_df = df[df["recorded_at"] >= cutoff_time].copy()

    print("Mốc chia:", cutoff_time)
    print("Train:", train_df.shape)
    print("Test :", test_df.shape)

    print("\nPhân bố nhãn Train:")
    print(train_df["diagnostic_state"].value_counts())

    print("\nPhân bố nhãn Test:")
    print(test_df["diagnostic_state"].value_counts())

    return train_df, test_df

def train_logistic_regression(train_df, test_df):
    print("\n===== LOGISTIC REGRESSION =====")

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            )
        ]
    )

    start_fit = time.perf_counter()

    model.fit(X_train, y_train)

    fit_time = time.perf_counter() - start_fit

    start_predict = time.perf_counter()

    y_pred = model.predict(X_test)

    inference_time = time.perf_counter() - start_predict

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    print("\nKết quả Logistic Regression:")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"Fit time       : {fit_time:.4f} giây")
    print(f"Inference time : {inference_time:.4f} giây")

    return model

def train_random_forest(train_df, test_df):
    print("\n===== RANDOM FOREST =====")

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    start_fit = time.perf_counter()

    model.fit(X_train, y_train)

    fit_time = time.perf_counter() - start_fit

    start_predict = time.perf_counter()

    y_pred = model.predict(X_test)

    inference_time = time.perf_counter() - start_predict

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    print("\nKết quả Random Forest:")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"Fit time       : {fit_time:.4f} giây")
    print(f"Inference time : {inference_time:.4f} giây")

    return model


def train_lightgbm(train_df, test_df):
    print("\n===== LIGHTGBM =====")

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    preprocessor = build_preprocessor(X_train)

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=100,
                    random_state=42,
                    verbosity=-1
                )
            )
        ]
    )

    start_fit = time.perf_counter()

    model.fit(X_train, y_train)

    fit_time = time.perf_counter() - start_fit

    start_predict = time.perf_counter()

    y_pred = model.predict(X_test)

    inference_time = time.perf_counter() - start_predict

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            digits=4
        )
    )

    print("\nKết quả LightGBM:")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Macro F1       : {macro_f1:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print(f"Fit time       : {fit_time:.4f} giây")
    print(f"Inference time : {inference_time:.4f} giây")

    return model

def load_sample_data(nrows=50000):
    print("Đang đọc dữ liệu mẫu...")

    df = pd.read_csv(
        RAW_DATA_PATH,
        nrows=nrows,
        parse_dates=["recorded_at"]
    )

    print("Đọc dữ liệu thành công!")
    print("Kích thước:", df.shape)

    print("\n5 dòng đầu:")
    print(df.head())

    return df

def inspect_sample_data(df):
    print("\n===== KIỂM TRA DỮ LIỆU MẪU =====")

    print("\nDanh sách cột:")
    print(df.columns.tolist())

    print("\nKiểu dữ liệu:")
    print(df.dtypes)

    print("\nCác cột có giá trị thiếu:")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    print("\nPhân bố diagnostic_state:")
    print(df["diagnostic_state"].value_counts())

    print("\nTỷ lệ diagnostic_state (%):")
    print(
        df["diagnostic_state"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    print("\nKhoảng thời gian của sample:")
    print("Từ:", df["recorded_at"].min())
    print("Đến:", df["recorded_at"].max())


def prepare_features(df):
    print("\n===== CHUẨN BỊ FEATURES =====")

    # Nhãn cần dự đoán
    y = df["diagnostic_state"]

    # Các cột không dùng trực tiếp làm feature ở baseline hiện tại
    columns_to_drop = [
        "diagnostic_state",
        "reading_id",
        "recorded_at",
        "sensor_id",
        "firmware_version"
    ]

    X = df.drop(columns=columns_to_drop)

    print("Số dòng X:", X.shape[0])
    print("Số feature hiện tại:", X.shape[1])

    print("\nDanh sách feature:")
    print(X.columns.tolist())

    print("\nKích thước y:", y.shape)

    return X, y

if __name__ == "__main__":
    print("Chạy Module Baseline Scikit-Learn...")

    df = load_sample_data()

    inspect_sample_data(df)

    X, y = prepare_features(df)

    preprocessor = build_preprocessor(X)

    X_transformed = test_preprocessor(preprocessor, X)

    train_df, test_df = temporary_time_split(df)

    logistic_model = train_logistic_regression(
        train_df,
        test_df
    )

    random_forest_model = train_random_forest(
        train_df,
        test_df
    )

    lightgbm_model = train_lightgbm(
        train_df,
        test_df
    )