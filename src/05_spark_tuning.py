"""
Thành viên 5: Huấn luyện Mô hình Phân tán trên Spark MLlib & Tuning Siêu tham số.
Nhiệm vụ:
- Load dữ liệu đã qua tiền xử lý (từ TV2) và Spark Feature Pipeline (từ TV4).
- Huấn luyện RandomForestClassifier phân tán (tối ưu hóa hyperparameters).
- Đánh giá trên tập Test.
- Lưu mô hình Random Forest tốt nhất để phục vụ dự đoán.
"""

import sys
import os
import argparse
from pathlib import Path

# Đảm bảo import được các module từ thư mục gốc
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Cấu hình HADOOP_HOME cho Windows
if os.name == "nt" and "HADOOP_HOME" not in os.environ:
    tools_hadoop = ROOT_DIR / "tools" / "hadoop"
    if (tools_hadoop / "bin" / "winutils.exe").exists():
        os.environ["HADOOP_HOME"] = str(tools_hadoop)
        os.environ["PATH"] = str(tools_hadoop / "bin") + os.pathsep + os.environ.get("PATH", "")

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

from src.utils.logger import ResourceLogger
from src.utils.spark_utils import SparkSessionManager

def main():
    parser = argparse.ArgumentParser(description="Thành viên 5: Spark MLlib Model Tuning")
    parser.add_argument("--train", type=str, default="data/processed/tv2/train_features.parquet", help="Đường dẫn tập Train")
    parser.add_argument("--test", type=str, default="data/processed/tv2/test_features.parquet", help="Đường dẫn tập Test")
    parser.add_argument("--pipeline", type=str, default="models/spark_feature_pipeline/spark_feature_pipeline_model", help="Đường dẫn Spark Pipeline của TV4")
    parser.add_argument("--output-model", type=str, default="models/spark_rf_best_model", help="Đường dẫn lưu mô hình tốt nhất")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" KHỞI ĐỘNG MODULE SPARK MODEL TUNING ")
    print(" Thành viên 5: Huấn luyện và Tối ưu hóa Siêu tham số ")
    print("=" * 70 + "\n")

    spark = SparkSessionManager.get_spark_session(
        app_name="IoT_Building_Diagnostic_SparkTuning",
        master="local[*]",
        log_level="WARN"
    )

    # 1. Nạp tập dữ liệu Train / Test (Do TV2 tạo ra)
    print(f"-> [1] Đang nạp dữ liệu Train từ: {args.train}")
    print(f"-> [1] Đang nạp dữ liệu Test từ: {args.test}")
    try:
        train_df = spark.read.parquet(args.train)
        test_df = spark.read.parquet(args.test)
    except Exception as e:
        print(f" LỖI: Không tìm thấy file dữ liệu Parquet. Vui lòng chạy file của TV2 trước. Chi tiết: {e}")
        return

    # 2. Nạp Spark Feature Pipeline (Do TV4 tạo ra)
    print(f"-> [2] Đang nạp Feature Pipeline từ: {args.pipeline}")
    try:
        feature_pipeline = PipelineModel.load(args.pipeline)
    except Exception as e:
        print(f" LỖI: Không tìm thấy Spark Pipeline. Vui lòng chạy file của TV4 trước. Chi tiết: {e}")
        return

    # Transform để lấy cột 'features' và 'label'
    print("-> Đang Transform dữ liệu qua Pipeline...")
    transformed_train = feature_pipeline.transform(train_df).cache()
    transformed_test = feature_pipeline.transform(test_df).cache()

    # Kích hoạt hành động count() để lưu DataFrame vào Memory (Cache) giúp Train nhanh hơn
    train_count = transformed_train.count()
    test_count = transformed_test.count()
    print(f"   + Tập Train: {train_count:,} dòng")
    print(f"   + Tập Test: {test_count:,} dòng\n")

    # 3. Cấu hình Model & GridSearch
    print("-> [3] Khởi tạo RandomForest và Lưới tham số (ParamGrid)...")
    rf = RandomForestClassifier(featuresCol="features", labelCol="label", seed=42)

    # Lưới tham số (Giảm maxDepth xuống để tránh lỗi OutOfMemory trên máy cá nhân)
    # maxDepth=15 tạo ra quá nhiều node (2^15) gây nổ RAM (Java heap space)
    paramGrid = (ParamGridBuilder()
                 .addGrid(rf.maxDepth, [7, 10])
                 .addGrid(rf.numTrees, [20, 40])
                 .build())

    evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")

    cv = CrossValidator(estimator=rf,
                        estimatorParamMaps=paramGrid,
                        evaluator=evaluator,
                        numFolds=3,
                        parallelism=1) # Đổi thành 1 để tiết kiệm RAM (tránh chạy song song nhiều mô hình)

    # 4. Huấn luyện và Đo lường tài nguyên
    print("\n-> [4] BẮT ĐẦU HUẤN LUYỆN CROSS-VALIDATION (Hãy kiên nhẫn, quá trình này có thể tốn vài phút)...")
    logger = ResourceLogger()
    logger.start()

    # Để tránh tràn RAM (OOM) triệt để trên máy cá nhân khi chạy CrossValidation,
    # chúng ta sẽ lấy mẫu ngẫu nhiên (sample) 15% dữ liệu Train để tìm ra bộ tham số tốt nhất.
    # (Trong môi trường cụm phân tán thật, ta sẽ bỏ dòng sample này đi).
    print("   * Đang trích xuất ngẫu nhiên 15% dữ liệu Train để Tuning chống tràn RAM...")
    sample_train = transformed_train.sample(fraction=0.15, seed=42).cache()
    
    cvModel = cv.fit(sample_train)

    res = logger.stop()
    print(f"\n [HOÀN TẤT] Thời gian Tuning: {res['execution_time_seconds']} giây | RAM đỉnh: {res['peak_ram_mb']} MB\n")

    # 5. Đánh giá Mô hình tốt nhất (Best Model)
    bestModel = cvModel.bestModel
    print("-> [5] Cấu hình Mô hình Tốt nhất (Best Model):")
    print(f"   + maxDepth: {bestModel.getOrDefault('maxDepth')}")
    print(f"   + numTrees: {bestModel.getOrDefault('numTrees')}")

    print("\n-> Đang dự đoán trên tập Test...")
    predictions = bestModel.transform(transformed_test)

    acc_evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
    accuracy = acc_evaluator.evaluate(predictions)
    f1_score = evaluator.evaluate(predictions)

    print(f"   => KẾT QUẢ TRÊN TẬP TEST:")
    print(f"      - Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"      - F1-Score : {f1_score:.4f}\n")

    # [BỔ SUNG] Trích xuất chỉ số độ quan trọng đặc trưng (Feature Importance) và lưu bảng kết quả Tuning
    import pandas as pd
    import matplotlib.pyplot as plt

    print("-> Đang lưu kết quả Tuning và biểu đồ Feature Importance...")
    os.makedirs("docs/logs", exist_ok=True)
    os.makedirs("docs/charts", exist_ok=True)

    # Lưu bảng kết quả Tuning siêu tham số
    try:
        params = cvModel.getEstimatorParamMaps()
        metrics = cvModel.avgMetrics
        tuning_res = []
        for p, m in zip(params, metrics):
            param_dict = {param.name: val for param, val in p.items()}
            param_dict['cv_f1_score'] = m
            tuning_res.append(param_dict)
        pd.DataFrame(tuning_res).to_csv("docs/logs/tuning_results.csv", index=False)
    except Exception as e:
        print(f" LỖI khi lưu bảng Tuning: {e}")

    # Trích xuất và vẽ biểu đồ Feature Importance
    try:
        importances = bestModel.featureImportances.toArray()
        try:
            attrs = transformed_train.schema["features"].metadata["ml_attr"]["attrs"]
            feature_names = []
            for attr_type in ["numeric", "binary", "nominal"]:
                if attr_type in attrs:
                    for attr in attrs[attr_type]:
                        feature_names.append((attr["idx"], attr["name"]))
            feature_names.sort(key=lambda x: x[0])
            feature_names = [x[1] for x in feature_names]
        except Exception:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
        fi_df = fi_df.sort_values(by="Importance", ascending=False).head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(fi_df["Feature"][::-1], fi_df["Importance"][::-1], color="#3B82F6")
        plt.xlabel("Mức độ ảnh hưởng (Feature Importance)")
        plt.title("Top 15 Đặc Trưng Quan Trọng Nhất (Spark MLlib Random Forest)")
        plt.tight_layout()
        plt.savefig("docs/charts/feature_importance.png", dpi=300)
        plt.close()
        print(" [OK] Đã lưu docs/logs/tuning_results.csv và docs/charts/feature_importance.png")
    except Exception as e:
        print(f" LỖI khi vẽ biểu đồ Feature Importance: {e}")


    # 6. Lưu mô hình tốt nhất
    print(f"-> [6] Đang lưu mô hình tốt nhất vào: {args.output_model}")
    bestModel.write().overwrite().save(args.output_model)
    print(" [OK] Lưu mô hình thành công.")

if __name__ == "__main__":
    main()
