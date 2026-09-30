"""
Module giám sát hiệu năng phần cứng cho các thực nghiệm Benchmark.
Theo dõi thời gian chạy, RAM tiêu thụ đỉnh (Peak RAM), và tốc độ suy luận (Inference Latency).
"""
import time
import os
import psutil
import threading

class BenchmarkTracker:
    def __init__(self, sample_interval: float = 0.05):
        self.sample_interval = sample_interval
        self.process = psutil.Process(os.getpid())
        self.peak_ram_mb = 0.0
        self.start_ram_mb = 0.0
        self.end_ram_mb = 0.0
        self.start_time = 0.0
        self.elapsed_time = 0.0
        self._monitoring = False
        self._monitor_thread = None

    def _get_ram_usage_mb(self) -> float:
        """Đo tổng RAM RSS của tiến trình cha và tất cả tiến trình con (bao gồm JVM của Spark nếu có)."""
        try:
            total_bytes = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                try:
                    total_bytes += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total_bytes / (1024 * 1024)
        except Exception:
            return 0.0

    def _monitor_loop(self):
        while self._monitoring:
            current_ram = self._get_ram_usage_mb()
            if current_ram > self.peak_ram_mb:
                self.peak_ram_mb = current_ram
            time.sleep(self.sample_interval)

    def start(self):
        """Bắt đầu đo lường."""
        self.start_ram_mb = self._get_ram_usage_mb()
        self.peak_ram_mb = self.start_ram_mb
        self.start_time = time.perf_counter()
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self) -> dict:
        """Dừng đo lường và trả về kết quả thống kê."""
        self.elapsed_time = time.perf_counter() - self.start_time
        self._monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=0.5)
        
        self.end_ram_mb = self._get_ram_usage_mb()
        if self.end_ram_mb > self.peak_ram_mb:
            self.peak_ram_mb = self.end_ram_mb
            
        return {
            "elapsed_seconds": round(self.elapsed_time, 3),
            "start_ram_mb": round(self.start_ram_mb, 2),
            "peak_ram_mb": round(self.peak_ram_mb, 2),
            "ram_increase_mb": round(max(0, self.peak_ram_mb - self.start_ram_mb), 2)
        }
