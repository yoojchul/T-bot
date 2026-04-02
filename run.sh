#!/bin/bash
# T-bot RAG Web Application 실행 스크립트

cd "$(dirname "$0")"

# 환경 변수 설정 (필요시 수정)
export FRIENDLI_TOKEN="flp_xxxx"
export DEEPINFRA_API_KEY="Hxxxxxxx"
export MILVUS_HOST="${MILVUS_HOST:-localhost}"
export MILVUS_PORT="${MILVUS_PORT:-19530}"
export MYSQL_HOST="${MYSQL_HOST:-127.0.0.1}"
export MYSQL_USER="${MYSQL_USER:-root}"
export MYSQL_PASSWORD="${MYSQL_PASSWORD:-password}"
export MYSQL_FILES_DIR="${MYSQL_FILES_DIR:-/var/lib/mysql-files}"

echo "T-bot RAG 서버 시작 중..."
echo "접속 주소: http://localhost:8000"
echo "환경: $MILVUS_HOST:$MILVUS_PORT (Milvus), $MYSQL_HOST (MySQL)"

/root/csv2db/bin/python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
