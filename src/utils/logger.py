"""
Module chứa các hàm tiện ích đo lường tài nguyên phần cứng.
Dùng để ghi nhận thời gian thực thi (Execution Time) và mức tiêu thụ RAM đỉnh (Peak RAM Usage) 
khi huấn luyện mô hình (để so sánh giữa Scikit-Learn và Spark MLlib).
"""
import time
import psutil
import os

class ResourceLogger:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.start_time = None
        self.start_ram = None

    def start(self):
        """Bắt đầu đo lường thời gian và RAM."""
        self.start_time = time.time()
        self.start_ram = self.process.memory_info().rss / (1024 * 1024) # Đổi sang MB

    def stop(self):
        """Kết thúc đo lường và trả về kết quả."""
        if self.start_time is None:
            raise ValueError("Phải gọi hàm start() trước khi gọi stop()")
        
        end_time = time.time()
        end_ram = self.process.memory_info().rss / (1024 * 1024)
        
        exec_time = end_time - self.start_time
        ram_used = end_ram - self.start_ram
        
        return {
            "execution_time_seconds": round(exec_time, 4),
            "ram_used_mb": round(ram_used, 4),
            "peak_ram_mb": round(end_ram, 4)
        }

if __name__ == "__main__":
    # Test thử logger
    logger = ResourceLogger()
    logger.start()
    time.sleep(1) # Mô phỏng thời gian chạy
    results = logger.stop()
    print("Test Logger Results:", results)
