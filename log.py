import os
import datetime
# -------------------------------------------------------
# 로그 기록 헬퍼
# -------------------------------------------------------
MYSQL_FILES_DIR = os.getenv("MYSQL_FILES_DIR", "/var/lib/mysql-files")

def write_log(db_name: str, message: str):
    if not db_name:
        return
    log_path = os.path.join(MYSQL_FILES_DIR, f"{db_name}.log")
    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] {message}\n")
    except Exception:
        pass


