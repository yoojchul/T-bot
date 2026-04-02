# -*- coding: utf-8 -*-
"""
End-to-end RAG: User Query -> Milvus Hybrid Search -> Table selection loop -> MySQL schema -> SQL -> Execute
LLM      : Friendli.ai serverless  deepseek-ai/DeepSeek-V3-0324
Embedding: deepinfra.com           BAAI/bge-m3
Milvus   : localhost:19530
MySQL    : localhost:3306
"""

import os
import json
import re
from typing import List, Dict, Any, Optional

from openai import OpenAI
import csv2recap

from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker
import mysql.connector

from langchain_community.memory.kg import ConversationKGMemory
from langchain_openai import ChatOpenAI
from log import write_log


# -------------------------------------------------------
# 1) Milvus Hybrid Search
# -------------------------------------------------------
class MilvusHybridSearcher:
    def __init__(self, uri: str, collection_name: str):
        self.client          = MilvusClient(uri=uri)
        self.collection_name = collection_name

    def hybrid_search_tables(
        self,
        query: str,
        limit: int = 10,
        exclude_filenames: Optional[set] = None,
        dense_weight: float = 0.3,
        sparse_weight: float = 0.7,
    ) -> List[Dict[str, Any]]:
        exclude_filenames = exclude_filenames or set()

        q_dense, q_sparse = csv2recap.generate_embeddings([query])

        req_dense = AnnSearchRequest(
            data=q_dense,
            anns_field="dense_vector",
            param={"metric_type": "L2", "params": {"nprobe": 10}},
            limit=limit,
        )
        req_sparse = AnnSearchRequest(
            data=q_sparse,
            anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {"drop_ratio_search": 0.2}},
            limit=limit,
        )

        ranker  = WeightedRanker(dense_weight, sparse_weight)
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[req_dense, req_sparse],
            ranker=ranker,
            limit=limit,
            output_fields=["filename", "text"],
        )

        hits = []
        for batch in results:
            for h in batch:
                row = {
                    "filename": h.get("entity", {}).get("filename"),
                    "text":     h.get("entity", {}).get("text"),
                    "score":    h.get("score"),
                }
                if row["filename"] and row["filename"] not in exclude_filenames:
                    hits.append(row)

        hits.sort(key=lambda x: (x["score"] is None, x["score"]), reverse=True)

        seen, uniq = set(), []
        for r in hits:
            fn = r["filename"]
            if fn in seen:
                continue
            seen.add(fn)
            uniq.append(r)
        return uniq


# -------------------------------------------------------
# 2) FriendliClient
# -------------------------------------------------------
class FriendliClient:
    def __init__(
        self,
        token: str = None,
        model: str = "deepseek-ai/DeepSeek-V3.1",
        timeout: int = 120,
    ):
        token = token or os.getenv("FRIENDLI_TOKEN", "YOUR_FRIENDLI_TOKEN")
        self.model  = model
        self.client = OpenAI(
            api_key=token,
            base_url="https://api.friendli.ai/serverless/v1",
            timeout=timeout,
        )

    def chat(self, messages: List[Dict[str, str]], temperature: float = 0.1) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=2048,
        )
        return completion.choices[0].message.content


# -------------------------------------------------------
# 3) JSON 파싱 유틸
# -------------------------------------------------------
def _extract_json_strict(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"Model output has no JSON object:\n{text}")
    return json.loads(m.group(0))


# -------------------------------------------------------
# 4) Prompt builders
# -------------------------------------------------------
def build_prompt_need_more_tables(
    user_query: str,
    selected_tables: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    tables_brief = [
        {"filename": t["filename"], "text": (t.get("text") or "")[:600]}
        for t in selected_tables
    ]
    system = (
        "You are a data engineer assistant. "
        "Given a user information need and currently selected MySQL tables (with short descriptions), "
        "decide whether additional tables are required to answer the query correctly.\n\n"
        "You MUST output ONLY valid JSON with the following schema:\n"
        "{\n"
        '  "need_more": boolean,\n'
        '  "reason": string,\n'
        '  "milvus_query": string\n'
        "}\n\n"
        "Rules:\n"
        "- If need_more is false, milvus_query MUST be an empty string.\n"
        "- If need_more is true, milvus_query should be a short Korean search query suitable for Milvus.\n"
        "- Do not mention tables that are already selected as missing.\n"
    )
    user = (
        f"USER_QUERY:\n{user_query}\n\n"
        f"SELECTED_TABLES:\n"
        f"{json.dumps(tables_brief, ensure_ascii=False, indent=2)}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_prompt_generate_mysql_sql(
    user_query: str,
    table_schemas: Dict[str, List[Dict[str, str]]],
) -> List[Dict[str, str]]:
    system = (
        "You are an expert MySQL query writer.\n"
        "Given a user query and table schemas, write a single MySQL SELECT statement that answers the query.\n\n"
        "You MUST output ONLY valid JSON with the following schema:\n"
        "{\n"
        '  "sql": string,\n'
        '  "notes": string\n'
        "}\n\n"
        "Rules (critical):\n"
        "- Output ONLY a SELECT query (no INSERT/UPDATE/DELETE/DDL).\n"
        "- Prefer explicit column names (avoid SELECT *).\n"
        "- Use LIMIT 200 unless the user explicitly asks for all rows.\n"
        "- If date filtering is implied (e.g., '2024년5월'), implement it robustly.\n"
        "- If you need to union multiple tables, do it carefully with aligned columns.\n"
    )
    user = (
        f"USER_QUERY:\n{user_query}\n\n"
        f"TABLE_SCHEMAS (table -> columns):\n"
        f"{json.dumps(table_schemas, ensure_ascii=False, indent=2)}\n"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


# -------------------------------------------------------
# 5) MySQL schema + execution
# -------------------------------------------------------
class MySQLRunner:
    def __init__(self, host, user, password, database, port=3306):
        self.conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database, port=port
        )

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass

    def describe_table(self, table: str) -> List[Dict[str, str]]:
        cur = self.conn.cursor(dictionary=True)
        cur.execute(f"DESCRIBE `{table}`")
        rows = cur.fetchall()
        cur.close()
        return rows

    def run_select(self, sql: str) -> List[Dict[str, Any]]:
        cur = self.conn.cursor(dictionary=True)
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows


def is_safe_select(sql: str) -> bool:
    s = sql.strip()
    if ";" in s[:-1]:
        return False
    lowered = re.sub(r"\s+", " ", s.lower())
    bad = ["insert ", "update ", "delete ", "drop ", "alter ",
           "create ", "truncate ", "grant ", "revoke "]
    if any(k in lowered for k in bad):
        return False
    return lowered.startswith("select ") or lowered.startswith("with ")


# -------------------------------------------------------
# 6) 모듈 레벨 KG 메모리 (대화 전체에서 공유)
# -------------------------------------------------------
_kg_memory: Optional[ConversationKGMemory] = None


def _get_kg_memory() -> ConversationKGMemory:
    global _kg_memory
    if _kg_memory is None:
        friendli_token = os.getenv("FRIENDLI_TOKEN", "YOUR_FRIENDLI_TOKEN")
        friendli_model = os.getenv("FRIENDLI_MODEL", "deepseek-ai/DeepSeek-V3.1")
        lc_llm = ChatOpenAI(
            api_key=friendli_token,
            base_url="https://api.friendli.ai/serverless/v1",
            model=friendli_model,
            temperature=0.1,
        )
        _kg_memory = ConversationKGMemory(llm=lc_llm, return_messages=False)
    return _kg_memory


# -------------------------------------------------------
# 7) Web API용 run_query 함수
# -------------------------------------------------------
def run_query(user_query: str, db_name: str, callback=None) -> str:
    """
    user_query에 대한 LLM 최종 답변(str)을 반환한다.
    ConversationKGMemory를 통해 모든 입출력이 저장되고 참조된다.
    callback(event_dict): 진행 이벤트 콜백 (optional)
      이벤트 타입:
        {"type": "log",             "message": str}
        {"type": "tables_selected", "tables": list}
        {"type": "sql_generated",   "sql": str, "notes": str}
    """
    milvus_uri     = os.getenv("MILVUS_URI",        "http://localhost:19530")
    friendli_token = os.getenv("FRIENDLI_TOKEN",    "YOUR_FRIENDLI_TOKEN")
    friendli_model = os.getenv("FRIENDLI_MODEL",    "deepseek-ai/DeepSeek-V3.1")
    mysql_host     = os.getenv("MYSQL_HOST",        "localhost")
    mysql_user     = os.getenv("MYSQL_USER")
    mysql_password = os.getenv("MYSQL_PASSWORD")
    mysql_port     = int(os.getenv("MYSQL_PORT",    "3306"))

    def log(msg):
        write_log(db_name, msg)

    searcher = MilvusHybridSearcher(uri=milvus_uri, collection_name=db_name)
    llm      = FriendliClient(token=friendli_token, model=friendli_model)
    db       = MySQLRunner(
        host=mysql_host, user=mysql_user,
        password=mysql_password, database=db_name, port=mysql_port
    )

    try:
        # 1) 초기 Milvus 검색
        log("관련 테이블 검색 중...")
        exclude      = set()
        initial_hits = searcher.hybrid_search_tables(user_query, limit=10, exclude_filenames=exclude)
        if not initial_hits:
            raise RuntimeError("Milvus에서 테이블을 찾을 수 없습니다. 데이터가 올바르게 업로드되었는지 확인하세요.")

        selected = [initial_hits[0]]
        exclude.add(initial_hits[0]["filename"])
        log(f"초기 테이블: {initial_hits[0]['filename']}")

        # 2) 테이블 보완 루프
        max_rounds = 5
        for round_idx in range(1, max_rounds + 1):
            msgs     = build_prompt_need_more_tables(user_query, selected)
            out      = llm.chat(msgs, temperature=0.1)
            decision = _extract_json_strict(out)

            need_more    = bool(decision.get("need_more", False))
            reason       = str(decision.get("reason", ""))
            milvus_query = str(decision.get("milvus_query", "") or "").strip()

            log(f"[Round {round_idx}] 추가 테이블 필요: {need_more} - {reason[:80]}")
            if not need_more:
                break

            # log missing. made by claude
            #milvus_query = milvus_query or user_query 
            if not milvus_query:
                log(f"[WARN] need_more=True but milvus_query empty. Fallback to original user_query.")
                milvus_query = user_query

            new_hits = searcher.hybrid_search_tables(milvus_query, limit=10, exclude_filenames=exclude)
            if not new_hits:
                log("추가 테이블 없음")
                break

            new_table = new_hits[0]
            selected.append(new_table)
            exclude.add(new_table["filename"])
            log(f"추가 테이블: {new_table['filename']}")
        else:
            log(f"최대 {max_rounds}라운드 도달")

        # 선택된 테이블 목록 로그에만 기록
        table_names = [t["filename"] for t in selected]
        log(f"선택된 테이블: {', '.join(table_names)}")

        # 3) MySQL 스키마 조회
        log("MySQL 스키마 조회 중...")
        table_schemas: Dict[str, List[Dict[str, str]]] = {}
        for t in selected:
            table = t["filename"]
            desc  = db.describe_table(table)
            table_schemas[table] = [
                {"Field": r.get("Field"), "Type": r.get("Type"), "Null": r.get("Null")}
                for r in desc
            ]

        # 4) SQL 생성
        log("SQL 생성 중...")
        msgs_sql = build_prompt_generate_mysql_sql(user_query, table_schemas)
        out_sql  = llm.chat(msgs_sql, temperature=0.1)
        sql_obj  = _extract_json_strict(out_sql)
        sql      = (sql_obj.get("sql")   or "").strip()
        notes    = (sql_obj.get("notes") or "").strip()

        log(f"생성된 SQL: {sql[:200]}")
        if notes:
            log(f"SQL 설명: {notes}")

        # 5) 안전성 검사 및 실행
        if not is_safe_select(sql):
            raise RuntimeError("SQL 안전성 검사 실패 (SELECT 문만 허용)")

        log("쿼리 실행 중...")
        rows = db.run_select(sql)

        # JSON 직렬화 가능하도록 변환
        if rows:
            columns  = list(rows[0].keys())
            row_data = [[str(r.get(c, "")) for c in columns] for r in rows[:200]]
        else:
            columns  = []
            row_data = []

        log(f"결과: {len(rows)}행")

        # 6) KG 메모리에서 관련 컨텍스트 로드 후 최종 답변 생성
        log("최종 답변 생성 중...")
        memory = _get_kg_memory()

        kg_vars    = memory.load_memory_variables({"input": user_query})
        kg_context = kg_vars.get("history", "").strip()

        rows_preview = row_data[:10]
        result_summary = (
            f"사용 테이블: {', '.join(table_names)}\n\n"
            f"실행 SQL:\n{sql}\n\n"
            f"SQL 설명: {notes}\n\n"
            f"컬럼: {columns}\n\n"
            f"결과 ({len(row_data)}행 중 최대 10행):\n"
            f"{json.dumps(rows_preview, ensure_ascii=False, indent=2)}"
        )

        system_content = (
            "당신은 데이터 분석 전문가 어시스턴트입니다. "
            "SQL 쿼리 결과를 바탕으로 사용자의 질문에 한국어로 명확하게 답변하세요."
        )
        if kg_context:
            system_content += f"\n\n[이전 대화 지식 그래프]\n{kg_context}"

        answer_msgs = [
            {"role": "system", "content": system_content},
            {
                "role": "user",
                "content": (
                    f"질문: {user_query}\n\n"
                    f"쿼리 결과:\n{result_summary}\n\n"
                    "위 결과를 바탕으로 질문에 대한 답변을 작성해 주세요."
                ),
            },
        ]

        final_answer = llm.chat(answer_msgs, temperature=0.3)

        log(f"답변: {final_answer}")

        # KG 메모리에 저장
        memory.save_context(
            {"input": user_query},
            {"output": final_answer},
        )

        log("답변 생성 완료")
        return final_answer

    finally:
        db.close()

