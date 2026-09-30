# Đồ Án Dữ Liệu Lớn (Big Data Project)
## Đề tài 11: IoT Building Diagnostics (Giám sát Chất lượng Không khí & Hệ thống Thông gió)

### 1. Giới thiệu đề tài
Hệ thống chẩn đoán trạng thái vận hành tòa nhà thông minh dựa trên 120 cảm biến tại 24 khu vực thuộc 6 tòa nhà trong suốt năm 2025.
- **Quy mô dữ liệu:** 2.100.000 bản ghi, 24 thuộc tính.
- **Nhiệm vụ:** Phân loại đa lớp trạng thái hệ thống (`normal`, `ventilation_issue`, `thermal_issue`).
- **Công nghệ cốt lõi:** Python, Apache Spark MLlib, Scikit-Learn, Streamlit.

### 2. Cấu trúc thư mục dự án
- `app/`: Thư mục Web Dashboard tương tác thời gian thực bằng Streamlit (Thành viên 6)
  - `app.py`: Giao diện chính của ứng dụng web.
  - `model_service.py`: Xử lý logic tải mô hình và dự đoán.
  - `sample_data.py`: Chứa dữ liệu kịch bản giả lập để test giao diện.
- `data/`:
  - `raw/`: Dữ liệu gốc 2.1 triệu dòng (lưu máy cá nhân, chặn bởi `.gitignore`).
  - `sample/`: Dữ liệu mẫu `sample_500_rows.csv` dùng để test nhanh.
  - `processed/`: Dữ liệu đã làm sạch và tập Train/Test theo chuỗi thời gian (TV1 và TV2).
- `docs/`:
  - `charts/`: Biểu đồ trực quan hóa Benchmark (Scalability, RAM, Confusion Matrix, Heatmap).
  - `logs/`: File log ghi nhận thời gian thực thi và tài nguyên phần cứng.
  - `references/`: Tài liệu đề tài, bảng phân công nhiệm vụ và kế hoạch.
- `models/`: Chứa artifact mô hình Spark Feature Pipeline (TV4), Spark MLlib sau tuning (TV5) và các mô hình Scikit-Learn (.joblib) của TV3.
- `notebooks/`: File Jupyter Notebook phục vụ EDA và chứng minh sự cố tràn $2^{20}$ dòng của Excel (TV1).
- `src/`: Mã nguồn kỹ thuật chia theo từng module của 6 thành viên:
  - `utils/`: Chứa các module tiện ích dùng chung (`logger.py`, `spark_utils.py`, `benchmark_tracker.py`,...).
  - `01_eda.py`: Khám phá dữ liệu lớn (EDA).
  - `02_feature_engineering.py` & `feature_engineering.py`: Kỹ nghệ đặc trưng vật lý IoT & xử lý missing values (TV2).
  - `03_baseline_sklearn.py`: Pipeline Scikit-Learn trên máy đơn.
  - `04_spark_pipeline.py`: Cấu hình Spark Session & Feature Pipeline phân tán.
  - `05_spark_tuning.py`: Huấn luyện & Tuning mô hình phân tán trên Spark MLlib.
  - `06_benchmark.py`: Scalability Benchmark đo lường đối đầu giữa 2 giải pháp.
- `tests/`: Chứa các bài kiểm thử tự động (Unit Test).
- `tools/`: Chứa các công cụ hỗ trợ hệ điều hành (ví dụ: `winutils.exe` và `hadoop.dll` cho Windows).

### 3. Hướng dẫn cài đặt & Khởi chạy

Để chạy dự án thành công, vui lòng thực hiện tuần tự theo luồng dữ liệu (Data Pipeline) dưới đây:

#### Bước 1: Cài đặt môi trường
```bash
pip install -r requirements.txt
```
> **⚠️ LƯU Ý DỮ LIỆU GỐC:** Nhóm không đẩy file dataset lên kho lưu trữ vì dung lượng quá lớn (313MB). Trước khi chạy code, vui lòng copy file `11_iot_building_diagnostic.csv` và dán vào bên trong thư mục `data/raw/` của dự án này.

#### Bước 2: Tiền xử lý & Kỹ nghệ Đặc trưng
Cắt tập Train/Test theo chuỗi thời gian và tính toán các công thức vật lý:
```bash
python src/02_feature_engineering.py --engine spark --input data/raw/11_iot_building_diagnostic.csv
```
*(Kết quả lưu tại `data/processed/tv2/` và ma trận tương quan tại `docs/charts/tv2/`)*

#### Bước 3: Xây dựng Spark Feature Pipeline
Khởi tạo cấu trúc VectorAssembler, StandardScaler để chuẩn bị cho Machine Learning:
```bash
python src/04_spark_pipeline.py --input data/raw/11_iot_building_diagnostic.csv --benchmark
```
*(Kết quả lưu tại `models/spark_feature_pipeline/`)*

#### Bước 4: Huấn luyện Baseline trên máy đơn
Chạy thử nghiệm Logistic Regression, Random Forest, LightGBM bằng Scikit-Learn:
```bash
python src/03_baseline_sklearn.py
```
*(Kết quả lưu tại `models/*.joblib` và log tại `docs/logs/tv3_baseline_results.json`)*

#### Bước 5: Huấn luyện Phân tán Big Data
Tối ưu hóa siêu tham số (Hyperparameter Tuning) với Spark MLlib:
```bash
python src/05_spark_tuning.py
```
*(Mô hình tốt nhất lưu tại `models/spark_rf_best_model`)*

#### Bước 6: Chạy Thực nghiệm Đối đầu Benchmark
Đo lường thời gian và RAM khi dữ liệu phình to, so sánh Spark vs Sklearn:
```bash
python src/06_benchmark.py
```
*(Biểu đồ đối đầu lưu tại `docs/charts/`)*

#### Bước 7: Khởi chạy Web Dashboard Demo
Khởi động giao diện giám sát tòa nhà thời gian thực:
```bash
streamlit run app/app.py
```