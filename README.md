# Đồ Án Dữ Liệu Lớn (Big Data Project)
## Đề tài 11: IoT Building Diagnostics (Giám sát Chất lượng Không khí & Hệ thống Thông gió)

### 1. Giới thiệu đề tài
Hệ thống chẩn đoán trạng thái vận hành tòa nhà thông minh dựa trên 120 cảm biến tại 24 khu vực thuộc 6 tòa nhà trong suốt năm 2025.
- **Quy mô dữ liệu:** 2.100.000 bản ghi, 24 thuộc tính.
- **Nhiệm vụ:** Phân loại đa lớp trạng thái hệ thống (`normal`, `ventilation_issue`, `thermal_issue`).
- **Công nghệ cốt lõi:** Python, Apache Spark MLlib, Scikit-Learn, Streamlit.

### 2. Cấu trúc thư mục dự án
- `app/`: Web Dashboard tương tác thời gian thực bằng Streamlit (`app.py`).
- `data/`:
  - `raw/`: Dữ liệu gốc 2.1 triệu dòng (lưu máy cá nhân, chặn bởi `.gitignore`).
  - `sample/`: Dữ liệu mẫu `sample_500_rows.csv` dùng để test nhanh.
  - `processed/`: Dữ liệu đã làm sạch và tập Train/Test theo chuỗi thời gian.
- `docs/`:
  - `charts/`: Biểu đồ trực quan hóa Benchmark (Scalability, RAM, Confusion Matrix).
  - `logs/`: File log ghi nhận thời gian thực thi và tài nguyên phần cứng.
  - `references/`: Tài liệu đề tài, bảng phân công nhiệm vụ và kế hoạch.
- `models/`: Chứa artifact mô hình Spark MLlib sau khi huấn luyện và tuning.
- `notebooks/`: File Jupyter Notebook phục vụ EDA và chứng minh sự cố tràn $2^{20}$ dòng của Excel.
- `src/`: Mã nguồn kỹ thuật chia theo từng module của 6 thành viên:
  - `utils/logger.py`: Module đo lường thời gian và RAM (`psutil`) dùng chung.
  - `01_eda.py`: Khám phá dữ liệu lớn (EDA).
  - `02_feature_engineering.py`: Kỹ nghệ đặc trưng vật lý IoT & xử lý missing values.
  - `03_baseline_sklearn.py`: Pipeline Scikit-Learn trên máy đơn.
  - `04_spark_pipeline.py`: Cấu hình Spark Session & Feature Pipeline phân tán.
  - `05_spark_tuning.py`: Huấn luyện & Tuning mô hình phân tán trên Spark MLlib.
  - `06_benchmark.py`: Scalability Benchmark đo lường đối đầu giữa 2 giải pháp.

### 3. Hướng dẫn cài đặt & Khởi chạy

#### Cài đặt môi trường
```bash
pip install -r requirements.txt
```
#### Chạy thử nghiệm mô hình Baseline (Scikit-Learn)
```bash
python src/03_baseline_sklearn.py
```

#### Chạy mô hình phân tán (Apache Spark MLlib)
```bash
python src/05_spark_tuning.py
```

#### Khởi chạy Web Dashboard Demo (Streamlit)
```bash
streamlit run app/app.py
```

---