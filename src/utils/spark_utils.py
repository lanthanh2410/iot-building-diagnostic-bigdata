"""
Module tiện ích quản trị Spark Session và Tối ưu hóa Big Data.
Thành viên 4: Kiến trúc Hạ tầng Phân tán & Spark Feature Pipeline.

Cung cấp:
- SparkSessionManager: Khởi tạo SparkSession tối ưu cho cả Local Multi-core và Docker Cluster.
- Các thiết lập tối ưu hiệu năng: Shuffle partitions, AQE, Kryo serializer, Memory Fraction.
- Tiện ích đo lường, Caching (MEMORY_AND_DISK), tối ưu Partitioning.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional, Dict, Any
import psutil

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Cấu hình HADOOP_HOME tự động cho hệ điều hành Windows
def setup_windows_hadoop_env():
    """Tự động nhận diện và cấu hình HADOOP_HOME trên Windows nếu có tools/hadoop."""
    if os.name == "nt" and "HADOOP_HOME" not in os.environ:
        project_root = Path(__file__).resolve().parents[2]
        tools_hadoop = project_root / "tools" / "hadoop"
        if (tools_hadoop / "bin" / "winutils.exe").exists():
            os.environ["HADOOP_HOME"] = str(tools_hadoop)
            os.environ["PATH"] = str(tools_hadoop / "bin") + os.pathsep + os.environ.get("PATH", "")

setup_windows_hadoop_env()

from pathlib import Path
from pyspark.sql import SparkSession, DataFrame
from pyspark.storagelevel import StorageLevel


class SparkSessionManager:
    """Quản lý khởi tạo và cấu hình SparkSession chuẩn hóa cho toàn dự án."""

    @staticmethod
    def get_system_specs() -> Dict[str, Any]:
        """Lấy thông số phần cứng của máy chủ/máy trạm."""
        logical_cores = os.cpu_count() or 4
        physical_cores = psutil.cpu_count(logical=False) or (logical_cores // 2)
        total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
        available_ram_gb = round(psutil.virtual_memory().available / (1024 ** 3), 2)

        return {
            "logical_cores": logical_cores,
            "physical_cores": physical_cores,
            "total_ram_gb": total_ram_gb,
            "available_ram_gb": available_ram_gb,
        }

    @classmethod
    def get_spark_session(
        cls,
        app_name: str = "IoT_Building_Diagnostic_SparkPipeline",
        master: Optional[str] = None,
        driver_memory: Optional[str] = None,
        executor_memory: Optional[str] = None,
        shuffle_partitions: Optional[int] = None,
        enable_aqe: bool = True,
        enable_kryo: bool = True,
        log_level: str = "WARN",
    ) -> SparkSession:
        """
        Khởi tạo SparkSession tối ưu hóa hiệu năng I/O và tính toán phân tán.

        Parameters:
        -----------
        app_name : str
            Tên ứng dụng hiển thị trên Spark Web UI (port 4040/8080).
        master : str, optional
            Spark Master URL. Nếu None, kiểm tra biến môi trường SPARK_MASTER_URL,
            nếu không có sẽ mặc định chạy 'local[*]'.
        driver_memory : str, optional
            Bộ nhớ cấp phát cho Driver (VD: '4g', '6g'). Tự động tính toán nếu None.
        executor_memory : str, optional
            Bộ nhớ cấp phát cho Executor (VD: '4g', '6g'). Tự động tính toán nếu None.
        shuffle_partitions : int, optional
            Số partition khi shuffle (groupby, join, sort). Mặc định tối ưu theo số cores
            thay vì 200 mặc định của Spark.
        enable_aqe : bool
            Kích hoạt Adaptive Query Execution (AQE) của Spark 3.x+.
        enable_kryo : bool
            Sử dụng KryoSerializer thay cho Java serializer mặc định (nhanh hơn 10x).
        log_level : str
            Mức ghi log Spark: "INFO", "WARN", "ERROR". Mặc định "WARN".
        """
        specs = cls.get_system_specs()
        cores = specs["logical_cores"]
        total_ram = specs["total_ram_gb"]

        # 1. Xác định Master URL
        if master is None:
            master = os.environ.get("SPARK_MASTER_URL", f"local[{cores}]")

        # 2. Tính toán tài nguyên bộ nhớ Driver / Executor an toàn
        if driver_memory is None:
            if total_ram >= 16:
                driver_memory = "6g"
            elif total_ram >= 8:
                driver_memory = "4g"
            else:
                driver_memory = "2g"

        if executor_memory is None:
            executor_memory = driver_memory

        # 3. Tính toán shuffle partitions tối ưu (tránh nghẽn 200 partition trên máy local)
        if shuffle_partitions is None:
            if "local" in master:
                # Với máy local, đặt shuffle partitions = 2 * cores để cân bằng pipeline
                shuffle_partitions = max(4, min(cores * 2, 48))
            else:
                # Với cluster, tùy thuộc vào số executor cores
                shuffle_partitions = max(16, cores * 2)

        # 4. Khởi tạo Builder với các tham số Big Data tối ưu
        builder = (
            SparkSession.builder.appName(app_name)
            .master(master)
            .config("spark.driver.memory", driver_memory)
            .config("spark.executor.memory", executor_memory)
            .config("spark.driver.maxResultSize", "2g")
            # Cấu hình Shuffle & Parallelism
            .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
            .config("spark.default.parallelism", str(cores * 2))
            # Quản lý bộ nhớ thực thi và đệm (Execution & Storage Memory)
            # 80% heap cho Spark; 30% trong đó cho Cache/Storage, 70% còn lại cho Shuffle/Join/Agg
            .config("spark.memory.fraction", "0.8")
            .config("spark.memory.storageFraction", "0.3")
            .config("spark.memory.offHeap.enabled", "false")
            # Nén I/O để tăng tốc truyền dữ liệu đĩa và mạng
            .config("spark.rdd.compress", "true")
            .config("spark.shuffle.compress", "true")
            .config("spark.shuffle.spill.compress", "true")
            .config("spark.sql.inMemoryColumnarStorage.compressed", "true")
            # Tối ưu đọc ghi định dạng Parquet/ORC
            .config("spark.sql.parquet.compression.codec", "snappy")
            .config("spark.sql.execution.arrow.pyspark.enabled", "true")
            # Cấu hình UI
            .config("spark.ui.enabled", "true")
            .config("spark.ui.port", "4040")
        )

        # 5. Cấu hình Kryo Serializer
        if enable_kryo:
            builder = (
                builder.config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                .config("spark.kryoserializer.buffer.max", "512m")
            )

        # 6. Cấu hình Adaptive Query Execution (AQE)
        if enable_aqe:
            builder = (
                builder.config("spark.sql.adaptive.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
                .config("spark.sql.adaptive.coalescePartitions.initialPartitionNum", str(shuffle_partitions))
                .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "67108864")  # 64MB
                .config("spark.sql.adaptive.skewJoin.enabled", "true")
            )

        spark = builder.getOrCreate()
        spark.sparkContext.setLogLevel(log_level)

        return spark

    @classmethod
    def get_cluster_info(cls, spark: SparkSession) -> Dict[str, Any]:
        """Trích xuất thông tin cấu hình runtime của Spark Cluster."""
        sc = spark.sparkContext
        conf = sc.getConf()

        return {
            "app_name": sc.appName,
            "spark_version": spark.version,
            "master": sc.master,
            "default_parallelism": sc.defaultParallelism,
            "shuffle_partitions": spark.conf.get("spark.sql.shuffle.partitions"),
            "driver_memory": conf.get("spark.driver.memory", "default"),
            "executor_memory": conf.get("spark.executor.memory", "default"),
            "aqe_enabled": spark.conf.get("spark.sql.adaptive.enabled", "false"),
            "serializer": conf.get("spark.serializer", "JavaSerializer"),
            "ui_web_url": sc.uiWebUrl,
        }

    @staticmethod
    def print_cluster_banner(spark: SparkSession, title: str = "SPARK CLUSTER CONFIGURATION") -> None:
        """In thông số cấu hình cụm Spark một cách trực quan ra console."""
        info = SparkSessionManager.get_cluster_info(spark)
        specs = SparkSessionManager.get_system_specs()

        border = "=" * 70
        print(border)
        print(f" {title.center(68)} ")
        print(border)
        print(f" • Spark Version       : {info['spark_version']}")
        print(f" • App Name            : {info['app_name']}")
        print(f" • Master URL          : {info['master']}")
        print(f" • Default Parallelism : {info['default_parallelism']}")
        print(f" • Shuffle Partitions  : {info['shuffle_partitions']}")
        print(f" • Driver Memory       : {info['driver_memory']}")
        print(f" • Adaptive QE (AQE)   : {info['aqe_enabled']}")
        print(f" • Serializer          : {info['serializer'].split('.')[-1]}")
        print(f" • Spark Web UI        : {info['ui_web_url']}")
        print("-" * 70)
        print(f" • Host CPU Cores      : {specs['logical_cores']} logical ({specs['physical_cores']} physical)")
        print(f" • Host Total RAM      : {specs['total_ram_gb']} GB (Available: {specs['available_ram_gb']} GB)")
        print(border + "\n")


def cache_dataframe(
    df: DataFrame,
    storage_level: StorageLevel = StorageLevel.MEMORY_AND_DISK,
    name: Optional[str] = None,
) -> DataFrame:
    """
    Cơ chế Caching an toàn cho Big Data:
    Sử dụng StorageLevel.MEMORY_AND_DISK để nếu dữ liệu 2.1 triệu dòng vượt quá RAM,
    các partition dư thừa sẽ được lưu xuống ổ đĩa cục bộ (disk spill) thay vì crash OOM.
    """
    cached_df = df.persist(storage_level)
    return cached_df


def unpersist_dataframe(df: Optional[DataFrame]) -> None:
    """Giải phóng an toàn DataFrame khỏi bộ nhớ đệm (RAM & Disk)."""
    if df is not None:
        try:
            df.unpersist(blocking=False)
        except Exception:
            pass


def get_optimal_partitions_count(record_count: int, num_cores: int) -> int:
    """
    Tính số partition tối ưu dựa trên quy mô dữ liệu và số CPU cores.
    Nguyên tắc:
    - Mỗi partition nên có kích thước khoảng 50.000 - 150.000 bản ghi (~15-30MB bộ nhớ)
    - Số partition nên là bội số của số cores (1x - 3x số cores) để tận dụng tối đa CPU pipeline.
    """
    # Tối thiểu 2 partition/core, tối đa dựa trên số dòng (~100k dòng/partition)
    core_based = num_cores * 2
    size_based = max(2, (record_count + 99999) // 100000)
    return max(core_based, size_based)
