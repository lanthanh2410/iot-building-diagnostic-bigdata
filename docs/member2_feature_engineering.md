# Thành viên 2: Kỹ nghệ đặc trưng và xử lý dữ liệu khuyết

## Kết quả đã chạy trên dữ liệu gốc

Nguồn: `data/raw/11_iot_building_diagnostic.csv` (2.100.000 bản ghi). Chia theo `recorded_at`: huấn luyện trước **2025-10-01** (1.572.480 bản ghi), kiểm thử từ ngày này (527.520 bản ghi). Chỉ tập huấn luyện được dùng để học median và tính tương quan. Kết quả dưới đây được chạy bằng Pandas trên toàn bộ dữ liệu, vì môi trường hiện tại chưa có PySpark.

| Đặc trưng vật lý | Công thức | Tương quan với `ventilation_issue` | Tương quan với `thermal_issue` |
| --- | --- | ---: | ---: |
| `filter_pressure_per_airflow` | `filter_pressure_pa / air_flow_m3_h` | 0,898 | -0,129 |
| `hvac_kw_per_airflow` | `hvac_power_kw / air_flow_m3_h` | 0,823 | 0,023 |
| `pm25_indoor_outdoor_ratio` | `pm25_ug_m3 / outdoor_pm25_ug_m3` | 0,603 | -0,081 |
| `delta_humidity_pct` | `indoor_humidity_pct - outdoor_humidity_pct` | 0,134 | 0,007 |
| `delta_temp_c` | `indoor_temp_c - outdoor_temp_c` | 0,014 | 0,105 |

Các hệ số là Pearson r giữa một đặc trưng liên tục và biến chỉ báo một-với-còn-lại của từng trạng thái. [Ma trận tương quan](charts/tv2/feature_correlation_matrix.svg) và [số liệu tương quan](charts/tv2/feature_correlation_matrix.csv) được xuất từ tập huấn luyện. Hai đặc trưng `filter_pressure_per_airflow` và `hvac_kw_per_airflow` tương quan với nhau ở mức **0,901**; nên thử bỏ từng đặc trưng trong mô hình ở bước đánh giá của thành viên 3/5. Hệ số tuyến tính thấp của đặc trưng nhiệt không chứng minh nó vô ích cho mô hình phi tuyến. Chưa thể khẳng định tập tối ưu khi chưa so sánh F1 bằng một tập validation theo thời gian nằm bên trong giai đoạn huấn luyện. Chỉ đánh giá một lần trên tập kiểm thử tháng 10–12 sau khi chốt đặc trưng và mô hình.

Sau xử lý, cả bốn cột cần điền khuyết và năm đặc trưng mới đều không còn giá trị khuyết trong dữ liệu này. Phân bố nhãn huấn luyện: `normal` 1.178.049, `ventilation_issue` 282.061, `thermal_issue` 112.370. Phân bố nhãn kiểm thử: tương ứng 399.175, 93.961, 34.384.

| Cột được điền khuyết | Thiếu ở train | Thiếu ở test |
| --- | ---: | ---: |
| `occupancy_count` | 14.235 | 4.761 |
| `tvoc_ppb` | 25.321 | 8.410 |
| `vibration_mm_s` | 28.279 | 9.690 |
| `window_open_pct` | 17.373 | 5.950 |

## Cách xử lý

- Điền bốn cột `occupancy_count`, `tvoc_ppb`, `vibration_mm_s`, `window_open_pct` bằng median riêng của `building_zone`. Tập kiểm thử và bản ghi dự đoán dùng median học từ tập huấn luyện, không tính lại từ dữ liệu tương lai. Khu vực chưa từng thấy hoặc một cột trống hoàn toàn ở khu vực đó dùng median toàn tập huấn luyện.
- PySpark dùng [`pyspark.ml.feature.Imputer(strategy="median")`](https://spark.apache.org/docs/3.5.8/api/python/reference/api/pyspark.ml.feature.Imputer.html) cho từng khu vực; đây là median xấp xỉ theo Spark. Pandas dùng median chính xác. Sau khi fit, Spark broadcast bảng median nhỏ rồi join một lần để xử lý DataFrame lớn.
- Phép chia có mẫu số bằng 0 hoặc âm trả về null, tránh vô cực. Trên dữ liệu hiện có, năm cột đầu ra không có null. Nếu dùng dữ liệu mới có mẫu số không hợp lệ, pipeline mô hình cần xử lý các null này.
- `MODEL_FEATURE_COLUMNS` là danh sách đầu vào số cho mô hình. `reading_id`, `recorded_at`, `sensor_id`, `firmware_version` và `diagnostic_state` không nằm trong danh sách này. `building_zone` cần được mã hóa riêng bởi pipeline của thành viên 3/4.

## Cách chạy

Từ gốc repository, sau khi cài `requirements.txt`:

```bash
python src/02_feature_engineering.py --engine spark --input data/raw/11_iot_building_diagnostic.csv
```

Nếu chưa cài PySpark nhưng có Pandas, chạy:

```bash
python src/02_feature_engineering.py --engine pandas --input data/raw/11_iot_building_diagnostic.csv
```

Để nhận checkpoint đã chia thời gian của thành viên 1, truyền `--train <path> --test <path>` thay cho `--input`. Đầu ra ở `data/processed/tv2/`: Spark ghi thư mục Parquet `train_features.parquet` và `test_features.parquet`; Pandas ghi hai file CSV cùng tiền tố. `data/processed/` nằm trong `.gitignore`, nên mỗi người cần chạy tạo checkpoint tại máy mình. Biểu đồ và số liệu nhỏ ở `docs/charts/tv2/` được giữ trong Git.

Trong mã Python, gọi `PandasZoneMedianFeatureEngineer().fit(train).transform(test)` hoặc `SparkZoneMedianFeatureEngineer().fit(train).transform(test)` từ `src.feature_engineering`. Dùng `fit_transform(train)` để tạo tập huấn luyện tương ứng. Giữ cùng một đối tượng đã fit cho cả hai tập.
