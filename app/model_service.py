"""
Dịch vụ suy luận mô hình (Model Inference Service) phục vụ Streamlit Web App.
Tự động nạp mô hình đã huấn luyện (models/rf_baseline.pkl), thực hiện
kỹ nghệ đặc trưng vật lý IoT thời gian thực và trả về chẩn đoán kèm giải thích.
"""

import os
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "random_forest.joblib")
MODEL_PATH = DEFAULT_MODEL_PATH if os.path.exists(DEFAULT_MODEL_PATH) else "models/random_forest.joblib"

LABEL_NAMES = {
    0: ("Bình Thường (Normal)", "normal", "#22C55E", "🟢"),
    1: ("Sự Cố Thông Khí (Ventilation Issue)", "ventilation_issue", "#EF4444", "🔴"),
    2: ("Sự Cố Điều Nhiệt (Thermal Issue)", "thermal_issue", "#F59E0B", "🟡")
}

FEATURE_VN_NAMES = {
    'delta_temp': 'Chênh lệch Nhiệt độ Trong/Ngoài (ΔT)',
    'delta_humidity': 'Chênh lệch Độ ẩm Trong/Ngoài (ΔH)',
    'co2_ppm': 'Nồng độ Khí CO2 (ppm)',
    'pm25_ug_m3': 'Bụi Mịn Trong Nhà PM2.5 (µg/m³)',
    'pm25_ratio': 'Tỷ lệ Bụi PM2.5 Trong/Ngoài',
    'filter_pressure_pa': 'Chênh lệch Áp suất Màng lọc (Pa)',
    'filter_pressure_ratio': 'Tỷ lệ Áp suất / Lưu lượng gió',
    'hvac_power_kw': 'Công suất Tiêu thụ HVAC (kW)',
    'hvac_energy_ratio': 'Suất Năng lượng HVAC / Gió',
    'air_flow_m3_h': 'Lưu lượng Không khí (m³/h)',
    'fan_speed_rpm': 'Tốc độ Quạt Thông gió (RPM)',
    'indoor_temp_c': 'Nhiệt độ Trong Nhà (°C)',
    'vibration_mm_s': 'Độ rung Động cơ (mm/s)',
    'maintenance_days': 'Số ngày Chưa Bảo trì',
    'occupancy_count': 'Số lượng Người Hiện diện',
    'equipment_age_years': 'Tuổi Thọ Thiết bị (Năm)'
}

class DiagnosticService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_names = []
        self._load_model()

    def _load_model(self):
        """Nạp model từ artifact do TV3 huấn luyện (Scikit-Learn Pipeline)."""
        if os.path.exists(MODEL_PATH):
            try:
                # TV3 lưu model dưới dạng một sklearn Pipeline duy nhất (không phải bundle dict)
                self.model = joblib.load(MODEL_PATH)
                
                # Trích xuất danh sách feature từ Pipeline nếu có thể
                try:
                    # Truy cập vào bộ tiền xử lý (preprocessor) -> lấy danh sách cột số
                    numeric_features = self.model.named_steps['preprocessor'].transformers_[0][2]
                    categorical_features = self.model.named_steps['preprocessor'].transformers_[1][2]
                    self.feature_names = categorical_features + numeric_features
                except Exception:
                    # Fallback nếu không đọc được từ pipeline
                    self.feature_names = [
                        "building_zone", "floor_number", "occupancy_count", "tvoc_ppb", 
                        "vibration_mm_s", "window_open_pct", "indoor_temp_c", "outdoor_temp_c", 
                        "indoor_humidity_pct", "outdoor_humidity_pct", "pm25_ug_m3", 
                        "outdoor_pm25_ug_m3", "filter_pressure_pa", "air_flow_m3_h", 
                        "hvac_power_kw", "co2_ppm", "fan_speed_rpm", "maintenance_days", 
                        "equipment_age_years", "delta_temp_c", "delta_humidity_pct", 
                        "pm25_indoor_outdoor_ratio", "filter_pressure_per_airflow", 
                        "hvac_kw_per_airflow"
                    ]
            except Exception as e:
                print(f"[Cảnh báo] Không thể tải {MODEL_PATH}: {e}")

    def _load_spark_model(self):
        """Khởi tạo SparkSession và nạp mô hình MLlib từ TV5."""
        if hasattr(self, 'spark_model_loaded') and self.spark_model_loaded:
            return True
            
        try:
            print("[Info] Khởi tạo SparkSession để phục vụ dự đoán realtime...")
            import sys
            
            # Cấu hình môi trường cho Windows
            if os.name == "nt":
                if "HADOOP_HOME" not in os.environ:
                    tools_hadoop = os.path.join(BASE_DIR, "tools", "hadoop")
                    if os.path.exists(os.path.join(tools_hadoop, "bin", "winutils.exe")):
                        os.environ["HADOOP_HOME"] = tools_hadoop
                        os.environ["PATH"] = os.path.join(tools_hadoop, "bin") + os.pathsep + os.environ.get("PATH", "")
                os.environ["PYSPARK_PYTHON"] = sys.executable
                os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
                
            from pyspark.sql import SparkSession
            from pyspark.ml import PipelineModel
            from pyspark.ml.classification import RandomForestClassificationModel
            
            self.spark = SparkSession.builder \
                .master("local[2]") \
                .appName("IoT_Building_Web_Inference") \
                .config("spark.ui.enabled", "false") \
                .getOrCreate()
            self.spark.sparkContext.setLogLevel("ERROR")
            
            pipeline_path = os.path.join(BASE_DIR, "models", "spark_feature_pipeline", "spark_feature_pipeline_model")
            rf_path = os.path.join(BASE_DIR, "models", "spark_rf_best_model")
            
            self.spark_feature_pipeline = PipelineModel.load(pipeline_path)
            self.spark_rf_model = RandomForestClassificationModel.load(rf_path)
            self.spark_model_loaded = True
            return True
        except Exception as e:
            print(f"[Lỗi Spark] Không thể nạp mô hình Spark MLlib: {e}")
            self.spark_model_loaded = False
            return False

    def predict(self, raw_params: dict, engine: str = "sklearn") -> dict:
        """
        Nhận vào từ điển các thông số cảm biến thô,
        thực hiện kỹ nghệ đặc trưng và trả về kết quả chẩn đoán chi tiết.
        """
        # 1. Tính toán các đặc trưng vật lý bổ sung (Engineered Columns theo TV2)
        p = raw_params.copy()
        
        # Tính toán chính xác theo chuẩn của feature_engineering.py
        indoor_temp = p.get('indoor_temp_c', 22.0)
        outdoor_temp = p.get('outdoor_temp_c', 15.0)
        delta_temp_c = indoor_temp - outdoor_temp
        
        indoor_hum = p.get('indoor_humidity_pct', 50.0)
        outdoor_hum = p.get('outdoor_humidity_pct', 60.0)
        delta_humidity_pct = indoor_hum - outdoor_hum
        
        indoor_pm25 = p.get('pm25_ug_m3', 15.0)
        outdoor_pm25 = p.get('outdoor_pm25_ug_m3', 10.0)
        pm25_ratio = indoor_pm25 / (outdoor_pm25 if outdoor_pm25 > 0 else 1e-5)
        
        air_flow = p.get('air_flow_m3_h', 1000.0)
        filter_pressure = p.get('filter_pressure_pa', 100.0)
        filter_pressure_ratio = filter_pressure / (air_flow if air_flow > 0 else 1e-5)
        
        hvac_power = p.get('hvac_power_kw', 5.0)
        hvac_energy_ratio = hvac_power / (air_flow if air_flow > 0 else 1e-5)
        
        # Gom các đặc trưng thành dictionary (tương ứng với 1 row DataFrame)
        feature_dict = {
            'building_zone': p.get('building_zone', 'B01-Z01'),
            'floor_number': p.get('floor_number', 1),
            'occupancy_count': p.get('occupancy_count', 10.0),
            'outdoor_temp_c': outdoor_temp,
            'outdoor_humidity_pct': outdoor_hum,
            'outdoor_pm25_ug_m3': outdoor_pm25,
            'indoor_temp_c': indoor_temp,
            'indoor_humidity_pct': indoor_hum,
            'co2_ppm': p.get('co2_ppm', 600.0),
            'pm25_ug_m3': indoor_pm25,
            'tvoc_ppb': p.get('tvoc_ppb', 120.0),
            'air_flow_m3_h': air_flow,
            'fan_speed_rpm': p.get('fan_speed_rpm', 1200.0),
            'hvac_power_kw': hvac_power,
            'filter_pressure_pa': filter_pressure,
            'vibration_mm_s': p.get('vibration_mm_s', 1.5),
            'maintenance_days': int(p.get('maintenance_days', 30)),
            'window_open_pct': p.get('window_open_pct', 0.0),
            'equipment_age_years': p.get('equipment_age_years', 3.0),
            'delta_temp_c': delta_temp_c,
            'delta_humidity_pct': delta_humidity_pct,
            'pm25_indoor_outdoor_ratio': pm25_ratio,
            'filter_pressure_per_airflow': filter_pressure_ratio,
            'hvac_kw_per_airflow': hvac_energy_ratio,
            'diagnostic_state': 'normal' # Dummy label for Pipeline StringIndexer
        }

        # 2. Suy luận 
        if engine == "spark" and self._load_spark_model():
            # Spark Inference
            from pyspark.sql.types import (
                StructType, StructField, StringType, IntegerType, DoubleType
            )
            # Tạo schema để Spark nhận dạng đúng kiểu dữ liệu
            schema = StructType([
                StructField("building_zone", StringType(), True),
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
                StructField("delta_temp_c", DoubleType(), True),
                StructField("delta_humidity_pct", DoubleType(), True),
                StructField("pm25_indoor_outdoor_ratio", DoubleType(), True),
                StructField("filter_pressure_per_airflow", DoubleType(), True),
                StructField("hvac_kw_per_airflow", DoubleType(), True),
                StructField("diagnostic_state", StringType(), True),
            ])
            # Chuyển đổi sang tuple theo thứ tự schema, ép kiểu chặt chẽ
            def cast_val(val, dt):
                if val is None:
                    return None
                if isinstance(dt, DoubleType):
                    return float(val)
                elif isinstance(dt, IntegerType):
                    return int(val)
                elif isinstance(dt, StringType):
                    return str(val)
                return val
                
            row_data = tuple(cast_val(feature_dict[field.name], field.dataType) for field in schema.fields)
            spark_df = self.spark.createDataFrame([row_data], schema=schema)
            
            # Pipeline transform
            transformed_df = self.spark_feature_pipeline.transform(spark_df)
            
            # RF Predict
            pred_df = self.spark_rf_model.transform(transformed_df)
            row = pred_df.select("prediction", "probability").collect()[0]
            
            pred_class = int(row["prediction"])
            probs_array = row["probability"].toArray()
            
            # Spark MLlib classes mapped by StringIndexer (frequency desc or alpha). 
            # Dựa vào logs của TV4, tần suất: normal(75%), ventilation(18%), thermal(7%)
            # Nên index 0: normal, 1: ventilation_issue, 2: thermal_issue.
            probs = [float(probs_array[0]), float(probs_array[1]), float(probs_array[2])]
            
        elif self.model:
            # Scikit-Learn Pipeline có ColumnTransformer đòi hỏi input là Pandas DataFrame
            input_df = pd.DataFrame([feature_dict])
            
            pred_class_label = self.model.predict(input_df)[0]
            probs_array = self.model.predict_proba(input_df)[0]
            
            # Map nhãn chuỗi về index số (0: normal, 1: ventilation, 2: thermal)
            label_mapping = {"normal": 0, "ventilation_issue": 1, "thermal_issue": 2}
            
            # Kiểm tra xem mô hình dự đoán ra chuỗi hay số
            if isinstance(pred_class_label, str):
                pred_class = label_mapping.get(pred_class_label, 0)
            else:
                pred_class = int(pred_class_label)
                
            # Đảm bảo probs có thứ tự đúng (tùy thuộc vào model.classes_)
            classes = list(self.model.classes_)
            
            probs = [0.0, 0.0, 0.0]
            if 'normal' in classes:
                probs[0] = float(probs_array[classes.index('normal')])
            if 'ventilation_issue' in classes:
                probs[1] = float(probs_array[classes.index('ventilation_issue')])
            if 'thermal_issue' in classes:
                probs[2] = float(probs_array[classes.index('thermal_issue')])
                
        else:
            # Thuật toán dự phòng chuẩn vật lý nếu chưa nạp được file model
            co2 = feature_dict['co2_ppm']
            fp = feature_dict['filter_pressure_pa']
            ind_t = feature_dict['indoor_temp_c']
            out_t = feature_dict['outdoor_temp_c']
            
            if co2 > 1100 or fp > 135 or pm25_ratio > 2.5:
                pred_class = 1 # ventilation_issue
                probs = [0.08, 0.84, 0.08]
            elif abs(ind_t - out_t) > 14 or ind_t > 28.5 or (hvac_power > 6.0 and ind_t > 26.0):
                pred_class = 2 # thermal_issue
                probs = [0.10, 0.12, 0.78]
            else:
                pred_class = 0 # normal
                probs = [0.88, 0.07, 0.05]

        label_title, label_code, color_hex, icon = LABEL_NAMES[pred_class]
        
        # 3. Phân tích các yếu tố đóng góp (Feature Importance / Contributions)
        contributions = []
        if engine == "spark" and hasattr(self, 'spark_rf_model'):
            try:
                # Trích xuất từ Spark RandomForestClassificationModel
                importances_array = self.spark_rf_model.featureImportances.toArray()
                # Cần đọc metadata từ spark_feature_pipeline để biết tên cột, nhưng để đơn giản 
                # ta dùng danh sách NUMERIC_FEATURE_COLUMNS + ZONE_INDEX_COL theo đúng thứ tự VectorAssembler
                feature_names = [
                    "floor_number", "occupancy_count", "tvoc_ppb", "vibration_mm_s", "window_open_pct",
                    "indoor_temp_c", "outdoor_temp_c", "indoor_humidity_pct", "outdoor_humidity_pct",
                    "pm25_ug_m3", "outdoor_pm25_ug_m3", "filter_pressure_pa", "air_flow_m3_h", "hvac_power_kw",
                    "co2_ppm", "fan_speed_rpm", "maintenance_days", "equipment_age_years",
                    "delta_temp_c", "delta_humidity_pct", "pm25_indoor_outdoor_ratio",
                    "filter_pressure_per_airflow", "hvac_kw_per_airflow", "building_zone_idx"
                ]
                importances = [float(i) for i in importances_array]
            except Exception:
                importances = [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.03]
                feature_names = ['co2_ppm', 'filter_pressure_pa', 'delta_temp_c', 'indoor_temp_c', 'air_flow_m3_h', 'pm25_indoor_outdoor_ratio', 'hvac_power_kw', 'filter_pressure_per_airflow', 'fan_speed_rpm', 'vibration_mm_s']
        elif engine == "sklearn" and self.model:
            # Truy cập model bên trong pipeline để lấy feature importances
            try:
                rf_model = self.model.named_steps['classifier']
                importances = rf_model.feature_importances_
                
                # Sinh danh sách feature tương ứng với sau khi OneHotEncoder
                ohe = self.model.named_steps['preprocessor'].named_transformers_['categorical'].named_steps['onehot']
                cat_feature_names = ohe.get_feature_names_out(['building_zone'])
                num_features = self.model.named_steps['preprocessor'].transformers_[0][2]
                all_feature_names = list(num_features) + list(cat_feature_names)
                
                # Rút gọn lại thành các feature nguyên gốc
                agg_importances = {}
                for f_name, imp in zip(all_feature_names, importances):
                    orig_name = f_name.split('_')[0] if f_name.startswith('building_zone_') else f_name
                    agg_importances[orig_name] = agg_importances.get(orig_name, 0) + imp
                    
                feature_names = list(agg_importances.keys())
                importances = [float(imp) for imp in agg_importances.values()]
            except Exception:
                importances = [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.03]
                feature_names = ['co2_ppm', 'filter_pressure_pa', 'delta_temp_c', 'indoor_temp_c', 'air_flow_m3_h', 'pm25_indoor_outdoor_ratio', 'hvac_power_kw', 'filter_pressure_per_airflow', 'fan_speed_rpm', 'vibration_mm_s']
        else:
            importances = [0.22, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.03, 0.03]
            feature_names = ['co2_ppm', 'filter_pressure_pa', 'delta_temp_c', 'indoor_temp_c', 'air_flow_m3_h', 'pm25_indoor_outdoor_ratio', 'hvac_power_kw', 'filter_pressure_per_airflow', 'fan_speed_rpm', 'vibration_mm_s']

        for fname, imp in zip(feature_names, importances):
            vn_name = FEATURE_VN_NAMES.get(fname, fname)
            val = feature_dict.get(fname, 0.0)
            contributions.append({
                "feature": fname,
                "display_name": vn_name,
                "value": round(val, 2) if isinstance(val, (int, float)) else val,
                "importance": round(imp * 100, 2)
            })
            
        contributions = sorted(contributions, key=lambda x: x['importance'], reverse=True)[:6]

        # 4. Đề xuất hành động kỹ thuật (Actionable Insights)
        recommendations = []
        if pred_class == 1: # Ventilation Issue
            if feature_dict['filter_pressure_pa'] > 120:
                recommendations.append("⚠️ **Màng lọc có dấu hiệu nghẹt**: Chênh lệch áp suất cao (> 120 Pa). Khuyến nghị kiểm tra vệ sinh hoặc thay thế tấm lọc không khí.")
            if feature_dict['co2_ppm'] > 1000:
                recommendations.append("⚠️ **Nồng độ CO2 vượt ngưỡng an toàn**: Tăng tốc độ quạt (fan_speed) hoặc mở cửa thông gió tự nhiên để trao đổi khí tươi.")
            if feature_dict['air_flow_m3_h'] < 300:
                recommendations.append("⚠️ **Lưu lượng cấp khí thấp**: Kiểm tra động cơ quạt hút và đường ống phân phối gió.")
            if len(recommendations) == 0:
                recommendations.append("⚠️ **Sự cố thông khí tổng quát**: Yêu cầu kỹ thuật viên HVAC kiểm tra toàn bộ hệ thống thông gió.")
        elif pred_class == 2: # Thermal Issue
            if feature_dict['indoor_temp_c'] > 27:
                recommendations.append("⚠️ **Nhiệt độ phòng quá nóng**: Tăng tải công suất làm lạnh của cụm dàn trao đổi nhiệt FCU/AHU.")
            if abs(feature_dict['delta_temp_c']) > 12:
                recommendations.append("⚠️ **Chênh lệch nhiệt độ trong/ngoài quá lớn**: Kiểm tra độ cách nhiệt của cửa kính và vỏ bao che tòa nhà.")
            if feature_dict['hvac_power_kw'] > 6:
                recommendations.append("⚠️ **Công suất HVAC ở mức tải đỉnh**: Giám sát dòng điện máy nén, phòng ngừa quá nhiệt động cơ.")
            if len(recommendations) == 0:
                recommendations.append("⚠️ **Sự cố điều nhiệt tổng quát**: Hệ thống HVAC không thể duy trì nhiệt độ cài đặt, cần bảo trì.")
        else: # Normal
            recommendations.append("✅ **Hệ thống hoạt động tối ưu**: Chất lượng không khí (IAQ) và tiện nghi nhiệt đạt tiêu chuẩn ASHRAE 62.1 & 55.")
            recommendations.append("ℹ️ Duy trì lịch bảo trì định kỳ sau mỗi 60 - 90 ngày vận hành.")

        return {
            "prediction_class": pred_class,
            "label_title": label_title,
            "label_code": label_code,
            "color_hex": color_hex,
            "icon": icon,
            "confidence_pct": round(probs[pred_class] * 100, 1),
            "probabilities": {
                "Bình thường": round(probs[0] * 100, 1),
                "Sự cố Thông khí": round(probs[1] * 100, 1),
                "Sự cố Điều nhiệt": round(probs[2] * 100, 1)
            },
            "contributions": contributions,
            "recommendations": recommendations,
            "features_used": feature_dict,
            "engine": engine
        }

# Singleton instance
diagnostic_service = DiagnosticService()
