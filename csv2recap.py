import os
import pandas as pd
import requests
import random
import csv
from openai import OpenAI
from pymilvus import (
        connections, FieldSchema, CollectionSchema, DataType,
        Collection, utility, AnnSearchRequest, WeightedRanker
)
import numpy as np
from scipy.sparse import csr_array
import time
from log import write_log

# --- Configuration ---
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

# --- API Keys ---
FRIENDLI_TOKEN    = os.getenv("FRIENDLI_TOKEN",    "YOUR_FRIENDLI_TOKEN")
DEEPINFRA_API_KEY = os.getenv("DEEPINFRA_API_KEY", "YOUR_DEEPINFRA_API_KEY")

# --- Friendli.ai 클라이언트 (OpenAI SDK 호환) ---
print(f"{FRIENDLI_TOKEN} == FRIENDLI_TOKEN")
friendli_client = OpenAI(
    api_key=FRIENDLI_TOKEN,
    base_url="https://api.friendli.ai/serverless/v1",
)
FRIENDLI_MODEL = "deepseek-ai/DeepSeek-V3.1"

DEEPINFRA_EMBED_URL = "https://api.deepinfra.com/v1/inference/BAAI/bge-m3-multi"


# -------------------------------------------------------
# Embedding: deepinfra.com BAAI/bge-m3  (dense + sparse)
# -------------------------------------------------------
def generate_embeddings(texts):
    headers = {
        "Authorization": f"Bearer {DEEPINFRA_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "inputs": texts,
        "dense": True,
        "sparse": True,
        "colbert": False
    }
    resp = requests.post(DEEPINFRA_EMBED_URL, json=payload, headers=headers, timeout=120)
    resp.raise_for_status()
    data = resp.json()

    dense_vectors = np.array(data["embeddings"], dtype=np.float32)
    raw_sparse    = data.get("sparse") or []
    sparse_vectors = [{i: val for i, val in enumerate(sv) if val != 0.0} for sv in raw_sparse]

    return dense_vectors, sparse_vectors


# -------------------------------------------------------
# LLM: Friendli.ai
# -------------------------------------------------------
def llm_generate(prompt: str, temperature: float = 0.3) -> str:
    completion = friendli_client.chat.completions.create(
        model=FRIENDLI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024,
    )
    return completion.choices[0].message.content


# -------------------------------------------------------
# Milvus 컬렉션 설정
# -------------------------------------------------------
def setup_milvus(collection_name):
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    fields = [
        FieldSchema(name="id",            dtype=DataType.INT64,          is_primary=True, auto_id=True),
        FieldSchema(name="filename",       dtype=DataType.VARCHAR,         max_length=256),
        FieldSchema(name="dense_vector",   dtype=DataType.FLOAT_VECTOR,    dim=1024),
        FieldSchema(name="sparse_vector",  dtype=DataType.SPARSE_FLOAT_VECTOR),
        FieldSchema(name="text",           dtype=DataType.VARCHAR,         max_length=2000),
    ]
    schema = CollectionSchema(fields, "File description embeddings")
    return Collection(collection_name, schema)


# -------------------------------------------------------
# CSV 스마트 읽기
# -------------------------------------------------------
def read_csv_smart(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as f:
        reader = csv.reader(f)
        try:
            header    = next(reader)
            col_count = len(header)
        except StopIteration:
            return pd.DataFrame()
        row_count = sum(1 for row in f)

    if col_count <= 10:
        limit_threshold = 100
    elif 10 < col_count <= 30:
        limit_threshold = 30
    else:
        limit_threshold = 10

    if row_count <= limit_threshold:
        return pd.read_csv(file_path, encoding=encoding)
    else:
        target_sample_size = int(row_count * 0.01)
        if target_sample_size < 1:
            target_sample_size = 1
        final_count      = min(target_sample_size, limit_threshold)
        indices_to_keep  = set(random.sample(range(1, row_count + 1), final_count))

        def skip_logic(x):
            if x == 0:
                return False
            return x not in indices_to_keep

        return pd.read_csv(file_path, skiprows=skip_logic, encoding=encoding)


# -------------------------------------------------------
# 메인 처리 함수
# -------------------------------------------------------
def recap_csv_files(directory, callback=None):
    """
    directory 내 CSV 파일을 Milvus에 벡터 임베딩으로 저장한다.
    callback(event_dict): 진행 이벤트를 전달하는 콜백 (optional)
      이벤트 타입: {"type": "log", "message": str}
    """

    milvus_db_name = os.path.basename(os.path.normpath(directory))

    def log(msg):
        write_log(milvus_db_name, msg)

    try:
        connections.connect("default", host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception:
        pass  # 이미 연결된 경우 무시

    milvus_col = setup_milvus(milvus_db_name)
    csv_files  = [f for f in os.listdir(directory) if f.endswith(".csv")]

    log(f"총 {len(csv_files)}개 CSV 파일 벡터화 시작")

    for filename in csv_files:
        file_path  = os.path.join(directory, filename)
        table_name = filename.replace(".csv", "")
        log(f"[{table_name}] 벡터화 중...")

        df_sample   = read_csv_smart(file_path)
        csv_snippet = df_sample.to_string()

        prompt = f"""{csv_snippet}\n\n {table_name} 이름으로된 csv의 일부이다.
                파일은 무엇을 담고 있는지 100자 내외로 설명하라.
                파일 이름에 date를 의미하는 부분이 포함될 수 있으니
                csv 파일 내용과 결부해서 date를 년월일을 구분해서 표기하라.
                모든 열의 헤더만 설명없이 나열하라. """

        response = llm_generate(prompt)
        log(f"[{table_name}] LLM 요약 완료")

        dense_vecs, sparse_vecs = generate_embeddings([response])
        entities = [
            [table_name],
            dense_vecs,
            sparse_vecs,
            [response[:2000]],
        ]
        milvus_col.insert(entities)
        log(f"[{table_name}] Milvus 저장 완료")

    milvus_col.flush()
    milvus_col.create_index("dense_vector",
            {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    milvus_col.create_index("sparse_vector",
            {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP",
             "params": {"drop_ratio_build": 0.2}})
    milvus_col.load()
    log("벡터 DB 인덱스 빌드 완료!")
