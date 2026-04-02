"""
T-bot RAG Web Application
FastAPI 백엔드 + SSE 기반 실시간 스트리밍
"""
import os
import sys
import json
import zipfile
import shutil
import threading
import asyncio
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

import mysql.connector
from pymilvus import connections, utility
from log import write_log

# -------------------------------------------------------
# 설정
# -------------------------------------------------------
MYSQL_FILES_DIR = os.getenv("MYSQL_FILES_DIR", "/var/lib/mysql-files")
MAX_UPLOAD_SIZE = 1 * 1024 * 1024 * 1024  # 1GB

MYSQL_CONFIG = {
    "user":               os.getenv("MYSQL_USER"),
    "password":           os.getenv("MYSQL_PASSWORD"),
    "host":               os.getenv("MYSQL_HOST",     "127.0.0.1"),
    "allow_local_infile": True,
}
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# -------------------------------------------------------
# FastAPI 앱
# -------------------------------------------------------
app = FastAPI(title="T-bot RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")



def make_callback(db_name: str, sync_cb):
    """로그 기록 + SSE 이벤트 전달 콜백 생성"""
    def callback(event):
        if event is None:
            sync_cb(None)
            return
        msg = event.get("message", "")
        if msg:
            write_log(db_name, msg)
        elif event.get("type") == "table_preview":
            write_log(db_name, f"[미리보기] {event.get('filename', '')}")
        elif event.get("type") == "sql_generated":
            write_log(db_name, f"[SQL] {event.get('sql', '')[:300]}")
        # type이 "log"인 이벤트는 로그 파일에만 기록하고 UI로 전송하지 않는다
        if event.get("type") == "log":
            return
        sync_cb(event)
    return callback


# -------------------------------------------------------
# SSE 스트리밍 헬퍼
# -------------------------------------------------------
async def run_in_thread_with_sse(fn, db_name: str):
    """동기 함수를 스레드에서 실행하고 SSE 이벤트를 스트리밍한다."""
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def sync_cb(event):
        try:
            loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception:
            pass

    cb = make_callback(db_name, sync_cb)

    thread = threading.Thread(target=fn, args=(cb,), daemon=True)
    thread.start()

    while True:
        try:
            event = await asyncio.wait_for(queue.get(), timeout=300.0)
        except asyncio.TimeoutError:
            yield f"data: {json.dumps({'type': 'error', 'message': '처리 시간 초과'}, ensure_ascii=False)}\n\n"
            break
        if event is None:
            break
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

    thread.join(timeout=5)


# -------------------------------------------------------
# 엔드포인트
# -------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = static_dir / "index.html"
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status(request: Request):
    db_name = request.cookies.get("database_name", "")
    return {"database": db_name}


@app.post("/api/upload")
async def upload_files(
    request: Request,
    response: Response,
    db_name: str = Form(...),
    files: List[UploadFile] = File(...),
):
    # 기존 데이터베이스 확인
    existing_db = request.cookies.get("database_name", "")
    if existing_db:
        return JSONResponse(
            status_code=400,
            content={"error": "존재하는 data가 있어 삭제가 필요합니다."},
        )

    # db_name 유효성 검사 (MySQL/Milvus 이름으로 안전한 문자만 허용)
    import re
    if not re.match(r'^[a-zA-Z0-9_]+$', db_name):
        return JSONResponse(
            status_code=400,
            content={"error": "데이터베이스 이름은 영문, 숫자, 언더스코어만 허용됩니다."},
        )

    # 파일 읽기 및 크기 확인
    total_size = 0
    file_contents = []
    for f in files:
        content = await f.read()
        total_size += len(content)
        file_contents.append((f.filename, content))

    if total_size > MAX_UPLOAD_SIZE:
        return JSONResponse(
            status_code=400,
            content={"error": f"총 파일 크기가 100GB를 초과합니다. ({total_size/1024/1024/1024:.2f}GB)"},
        )

    # 저장 디렉토리 생성
    dest_dir = os.path.join(MYSQL_FILES_DIR, db_name)
    os.makedirs(dest_dir, exist_ok=True)

    saved_files = []
    for filename, content in file_contents:
        if filename.lower().endswith(".zip"):
            # ZIP 압축 해제
            zip_path = os.path.join(dest_dir, filename)
            with open(zip_path, "wb") as f:
                f.write(content)
            with zipfile.ZipFile(zip_path, "r") as z:
                for member in z.namelist():
                    if member.lower().endswith(".csv") and not member.startswith("__"):
                        z.extract(member, dest_dir)
                        # 중첩 디렉토리 내 파일은 dest_dir로 이동
                        extracted = os.path.join(dest_dir, member)
                        if os.path.dirname(member):
                            base = os.path.basename(member)
                            target = os.path.join(dest_dir, base)
                            shutil.move(extracted, target)
                            saved_files.append(base)
                        else:
                            saved_files.append(member)
            os.remove(zip_path)
        elif filename.lower().endswith(".csv"):
            # 줄끝 정규화 (CRLF → LF)
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError:
                try: 
                    text = content.decode("utf-8-sig")
                except UnicodeDecodeError:
                    text = content.decode("cp949")
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            file_path = os.path.join(dest_dir, filename)
            with open(file_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(text)
            saved_files.append(filename)

    if not saved_files:
        shutil.rmtree(dest_dir, ignore_errors=True)
        return JSONResponse(
            status_code=400,
            content={"error": "업로드 가능한 CSV 파일이 없습니다."},
        )

    # 로그 초기화
    write_log(db_name, f"=== 데이터베이스 '{db_name}' 생성 ===")
    write_log(db_name, f"업로드된 파일: {', '.join(saved_files)}")

    # 쿠키 설정
    response.set_cookie(
        key="database_name",
        value=db_name,
        max_age=30 * 24 * 3600,
        samesite="lax",
    )

    return {"status": "uploaded", "db_name": db_name, "files": saved_files}


@app.get("/api/process")
async def process_stream(request: Request, db_name: str = ""):
    if not db_name:
        db_name = request.cookies.get("database_name", "")
    if not db_name:
        return JSONResponse(status_code=400, content={"error": "데이터베이스 없음"})

    directory = os.path.join(MYSQL_FILES_DIR, db_name)
    if not os.path.isdir(directory):
        return JSONResponse(status_code=400, content={"error": f"디렉토리 없음: {directory}"})

    async def generator():
        import csv2recap
        import csv2mysql

        def run(cb):
            try:
                cb({"type": "hourglass_start", "message": "벡터 DB 생성 중... (시간이 걸립니다)"})
                csv2recap.recap_csv_files(directory, callback=cb)
                cb({"type": "hourglass_end", "message": "벡터 DB 생성 완료"})

                cb({"type": "mysql_start", "message": "MySQL 데이터 로딩 시작..."})
                csv2mysql.process_directory(directory, callback=cb)
                cb({"type": "done", "message": "모든 처리가 완료되었습니다!"})
            except Exception as e:
                cb({"type": "error", "message": str(e)})
            finally:
                cb(None)

        async for chunk in run_in_thread_with_sse(run, db_name):
            yield chunk

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/query")
async def query_stream(request: Request):
    body       = await request.json()
    user_query = body.get("query", "").strip()
    db_name    = request.cookies.get("database_name", "")

    if not db_name:
        return JSONResponse(status_code=400, content={"error": "데이터베이스가 없습니다. 먼저 파일을 업로드하세요."})
    if not user_query:
        return JSONResponse(status_code=400, content={"error": "질문을 입력하세요."})

    write_log(db_name, f"\n=== USER QUERY: {user_query} ===")

    async def generator():
        import search

        def run(cb):
            try:
                result = search.run_query(user_query, db_name, callback=cb)
                #cb({"type": "result", **result})
                cb({"type": "done", "message": result})
            except Exception as e:
                cb({"type": "error", "message": str(e)})
            finally:
                cb(None)

        async for chunk in run_in_thread_with_sse(run, db_name):
            yield chunk

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/api/database")
async def delete_database(request: Request, response: Response):
    db_name = request.cookies.get("database_name", "")
    if not db_name:
        return JSONResponse(status_code=400, content={"error": "쿠키에 데이터베이스 정보가 없습니다."})

    errors = []

    # Milvus collection 삭제
    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
        if utility.has_collection(db_name):
            utility.drop_collection(db_name)
            write_log(db_name, f"Milvus collection '{db_name}' 삭제됨")
    except Exception as e:
        errors.append(f"Milvus: {e}")

    # MySQL database 삭제
    try:
        conn   = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"DROP DATABASE IF EXISTS `{db_name}`")
        conn.commit()
        cursor.close()
        conn.close()
        write_log(db_name, f"MySQL database '{db_name}' 삭제됨")
    except Exception as e:
        errors.append(f"MySQL: {e}")

    # 파일 디렉토리 삭제
    try:
        dest_dir = os.path.join(MYSQL_FILES_DIR, db_name)
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)
    except Exception as e:
        errors.append(f"파일 삭제: {e}")

    # 쿠키 삭제
    response.delete_cookie("database_name")

    if errors:
        return {"status": "partial", "errors": errors}
    return {"status": "deleted", "db_name": db_name}


# -------------------------------------------------------
# 실행
# -------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
