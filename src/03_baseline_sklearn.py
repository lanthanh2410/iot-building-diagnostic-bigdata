"""
Thành viên 3: Xây dựng & Đánh giá Mô hình Baseline Đơn máy (Scikit-Learn).
Nhiệm vụ:
- Huấn luyện Logistic Regression / Random Forest / LightGBM.
- Đánh giá Accuracy, F1-score.
- Log tài nguyên RAM/CPU.
"""

import os
import sys
import time
import json
import threading
import joblib
from pathlib import Path

import pandas as pd
import psutil


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from lightgbm import LGBMClassifier


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.feature_engineering import MODEL_FEATURE_COLUMNS


def get_process_memory_mb():
    """Lấy tổng RAM hiện tại của process Python và các child process."""
    process = psutil.Process(os.getpid())

    total_rss = process.memory_info().rss

    for child in process.children(recursive=True):
        try:
            total_rss += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    return total_rss / (1024 ** 2)


def monitor_peak_memory(stop_event, result, interval=0.05):
    """Theo dõi RAM liên tục và ghi nhận mức cao nhất."""
    peak_ram = get_process_memory_mb()

    while not stop_event.is_set():
        current_ram = get_process_memory_mb()

        if current_ram > peak_ram:
            peak_ram = current_ram

        time.sleep(interval)

    # Đo thêm lần cuối trước khi kết thúc
    peak_ram = max(peak_ram, get_process_memory_mb())

    result["peak_ram_mb"] = peak_ram


def build_preprocessor():
    print("\n===== XÂY DỰNG PREPROCESSING PIPELINE =====")

    categorical_features = ["building_zone"]

    numeric_features = list(MODEL_FEATURE_COLUMNS)

    numeric_transformer = Pipeline(
        steps=[
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

TRAIN_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "tv2",
    "train_features.csv"
)

TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "tv2",
    "test_features.csv"
)


def load_processed_data(train_rows=200000, test_rows=50000):
    print("\n===== ĐỌC DỮ LIỆU ĐÃ XỬ LÝ CỦA TV2 =====")

    train_df = pd.read_csv(
        TRAIN_DATA_PATH,
        nrows=train_rows,
        parse_dates=["recorded_at"]
    )

    test_df = pd.read_csv(
        TEST_DATA_PATH,
        nrows=test_rows,
        parse_dates=["recorded_at"]
    )

    print("Train:", train_df.shape)
    print("Test :", test_df.shape)

    return train_df, test_df


def train_logistic_regression(train_df, test_df):
    print("\n===== LOGISTIC REGRESSION =====")

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    preprocessor = build_preprocessor()

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

    ram_before = get_process_memory_mb()

    stop_event = threading.Event()
    memory_result = {}

    memory_thread = threading.Thread(
        target=monitor_peak_memory,
        args=(stop_event, memory_result),
        daemon=True
    )

    memory_thread.start()

    start_fit = time.perf_counter()

    model.fit(X_train, y_train)

    fit_time = time.perf_counter() - start_fit

    stop_event.set()
    memory_thread.join()

    peak_ram = memory_result["peak_ram_mb"]
    ram_increase = max(0.0, peak_ram - ram_before)

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

    print(f"Peak RAM       : {peak_ram:.2f} MB")
    print(f"RAM increase   : {ram_increase:.2f} MB")

    result = {
        "model": "Logistic Regression",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "fit_time_seconds": float(fit_time),
        "inference_time_seconds": float(inference_time),
        "peak_ram_mb": float(peak_ram),
        "ram_increase_mb": float(ram_increase)
    }

    return model, result

def train_random_forest(train_df, test_df):
    print("\n===== RANDOM FOREST =====")

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    preprocessor = build_preprocessor()

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

    ram_before = get_process_memory_mb()

    stop_event = threading.Event()
    memory_result = {}

    memory_thread = threading.Thread(
        target=monitor_peak_memory,
        args=(stop_event, memory_result),
        daemon=True
    )

    memory_thread.start()

    start_fit = time.perf_counter()

    model.fit(X_train, y_train)

    fit_time = time.perf_counter() - start_fit

    stop_event.set()
    memory_thread.join()

    peak_ram = memory_result["peak_ram_mb"]
    ram_increase = max(0.0, peak_ram - ram_before)

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

    print(f"Peak RAM       : {peak_ram:.2f} MB")
    print(f"RAM increase   : {ram_increase:.2f} MB")

    result = {
        "model": "Random Forest",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "fit_time_seconds": float(fit_time),
        "inference_time_seconds": float(inference_time),
        "peak_ram_mb": float(peak_ram),
        "ram_increase_mb": float(ram_increase)
    }

    return model, result


def train_lightgbm(train_df, test_df):
    print("\n===== LIGHTGBM =====")

    X_train, y_train = prepare_features(train_df)
    X_test, y_test = prepare_features(test_df)

    preprocessor = build_preprocessor()

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

    ram_before = get_process_memory_mb()

    stop_event = threading.Event()
    memory_result = {}

    memory_thread = threading.Thread(
        target=monitor_peak_memory,
        args=(stop_event, memory_result),
        daemon=True
    )

    memory_thread.start()

    start_fit = time.perf_counter()

    model.fit(X_train, y_train)

    fit_time = time.perf_counter() - start_fit

    stop_event.set()
    memory_thread.join()

    peak_ram = memory_result["peak_ram_mb"]
    ram_increase = max(0.0, peak_ram - ram_before)

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

    print(f"Peak RAM       : {peak_ram:.2f} MB")
    print(f"RAM increase   : {ram_increase:.2f} MB")

    result = {
        "model": "LightGBM",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "fit_time_seconds": float(fit_time),
        "inference_time_seconds": float(inference_time),
        "peak_ram_mb": float(peak_ram),
        "ram_increase_mb": float(ram_increase)
    }

    return model, result


def prepare_features(df):
    print("\n===== CHUẨN BỊ FEATURES =====")

    feature_columns = [
        "building_zone",
        *MODEL_FEATURE_COLUMNS
    ]

    X = df[feature_columns].copy()
    y = df["diagnostic_state"].copy()

    print("Số dòng X:", X.shape[0])
    print("Số feature hiện tại:", X.shape[1])

    print("\nDanh sách feature:")
    print(X.columns.tolist())

    print("\nKích thước y:", y.shape)

    return X, y

def save_benchmark_results(results):
    log_dir = os.path.join(
        BASE_DIR,
        "docs",
        "logs"
    )

    os.makedirs(log_dir, exist_ok=True)

    output_path = os.path.join(
        log_dir,
        "tv3_baseline_results.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4
        )

    print("\n===== LƯU BENCHMARK =====")
    print("Đã lưu:", output_path)

def save_models(
    logistic_model,
    random_forest_model,
    lightgbm_model
):
    model_dir = os.path.join(
        BASE_DIR,
        "models"
    )

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    model_paths = {
        "Logistic Regression": os.path.join(
            model_dir,
            "logistic_regression.joblib"
        ),
        "Random Forest": os.path.join(
            model_dir,
            "random_forest.joblib"
        ),
        "LightGBM": os.path.join(
            model_dir,
            "lightgbm.joblib"
        ),
    }

    joblib.dump(
        logistic_model,
        model_paths["Logistic Regression"]
    )

    joblib.dump(
        random_forest_model,
        model_paths["Random Forest"]
    )

    joblib.dump(
        lightgbm_model,
        model_paths["LightGBM"]
    )

    print("\n===== LƯU MODEL =====")

    for model_name, model_path in model_paths.items():
        print(f"{model_name}: {model_path}")



if __name__ == "__main__":
    print("Chạy Module Baseline Scikit-Learn...")

    train_df, test_df = load_processed_data()

    print("\nTrain range:")
    print(
        train_df["recorded_at"].min(),
        "->",
        train_df["recorded_at"].max()
    )

    print("\nTest range:")
    print(
        test_df["recorded_at"].min(),
        "->",
        test_df["recorded_at"].max()
    )

    logistic_model, logistic_result = train_logistic_regression(
        train_df,
        test_df
    )

    random_forest_model, random_forest_result = train_random_forest(
        train_df,
        test_df
    )

    lightgbm_model, lightgbm_result = train_lightgbm(
        train_df,
        test_df
    )
    benchmark_results = [
        logistic_result,
        random_forest_result,
        lightgbm_result
    ]

    save_benchmark_results(
        benchmark_results
    )

    save_models(
        logistic_model,
        random_forest_model,
        lightgbm_model
    )
