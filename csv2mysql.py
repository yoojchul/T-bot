import os
import pandas as pd
import mysql.connector
import re
from openai import OpenAI
import time
from log import write_log

# --- API Keys ---
FRIENDLI_TOKEN = os.getenv("FRIENDLI_TOKEN", "YOUR_FRIENDLI_TOKEN")

# --- Friendli.ai 클라이언트 (OpenAI SDK 호환) ---
friendli_client = OpenAI(
    api_key=FRIENDLI_TOKEN,
    base_url="https://api.friendli.ai/serverless/v1",
)
FRIENDLI_MODEL = "deepseek-ai/DeepSeek-V3.1"
#FRIENDLI_MODEL = "deepseek-ai/DeepSeek-V324"


# -------------------------------------------------------
# LLM
# -------------------------------------------------------
def llm_generate(prompt: str, temperature: float = 0.0) -> str:
    completion = friendli_client.chat.completions.create(
        model=FRIENDLI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=256,
    )
    return completion.choices[0].message.content


# -------------------------------------------------------
# resolve_token_type
# -------------------------------------------------------
def resolve_token_type(input_str: str):
    if not input_str:
        return None

    token_pattern = re.compile(
        r'(int|double|float|decimal|varchar|text|date|datetime|time|timestamp)(?:\((.*?)\))?',
        re.IGNORECASE
    )
    matches = list(token_pattern.finditer(input_str))

    last_end     = 0
    parsed_tokens = []

    for m in matches:
        start, end = m.span()
        if input_str[last_end:start].strip() != "":
            return None
        t_type  = m.group(1).lower()
        t_param = m.group(2) if m.group(2) else ""
        parsed_tokens.append((t_type, t_param))
        last_end = end

    if input_str[last_end:].strip() != "":
        return None
    if not parsed_tokens:
        return None

    type_set       = set(t[0] for t in parsed_tokens)
    numeric_group  = {'int', 'double', 'float', 'decimal'}
    varchar_group  = {'varchar'}
    text_group     = {'text'}
    datetime_group = {'date', 'datetime', 'time', 'timestamp'}

    if type_set.issubset(numeric_group):
        if 'double' in type_set:
            return 'double'
        if 'float' in type_set or 'decimal' in type_set:
            return 'float'
        return 'int'

    elif type_set.issubset(varchar_group):
        max_len = 0
        for _, param in parsed_tokens:
            if param.isdigit():
                max_len = max(max_len, int(param))
        return f"varchar({max_len})"

    elif type_set.issubset(text_group):
        return 'text'

    elif type_set.issubset(datetime_group):
        base_format = parsed_tokens[0][1]
        for _, param in parsed_tokens:
            if param != base_format:
                return None
        if len(type_set) == 1:
            final_type = list(type_set)[0]
        else:
            final_type = 'datetime'
        if base_format:
            return f"{final_type}({base_format})"
        else:
            return final_type

    else:
        return None


# --- MySQL Configuration ---
MYSQL_CONFIG = {
    'user':             os.getenv("MYSQL_USER"),
    'password':         os.getenv("MYSQL_PASSWORD"),
    'host':             os.getenv("MYSQL_HOST"),
    'allow_local_infile': True,
}


def get_optimal_types(df, callback=None):
    """LLM에게 20줄을 보내 MySQL 타입을 결정받는다."""
    columns  = df.columns.tolist()
    results  = []
    var_name = "@temp"
    fields   = "("
    set_stm  = ""

    for i in range(len(columns)):
        time.sleep(0.5)
        prompt = (
            f"{df.iloc[:, i].to_string(header=False, index=False)}.\n"
            f"문자열들은 csv file의 한 열이다. {columns[i]}는 이들 문자열의 제목인데 "
            f"Mysql로 변환할 때 적당한 타입만 표시하라. 설명이나 제목을 붙이지 마라."
            f"제목에 연월일, 시간, date 등이 포함되면 타입으로 DATE, DATETIME, TIME등을 사용하라. "
            f"date, datetime, time, timestamp를 선택할 때는 문자열들을 근거로 년,월, 일 순서를 파악하고 "
            f"타입 다음에 시간 형식도 같이 출력하라. "
            f"연도월일 순서이면 (%Y%m%d), 일이 없으면 (%Y%m), "
            f"연도-월-일 순이면 (%Y-%m-%d)를 출력하고, 월-일-년도 순서이면 (%m-%d-%Y) 로 출력한다. "
            f"추가로 primary, field 이름, 설명이나 comment는 넣지 마라. "
            f"VARCHAR type은 반드시 크기를 지정하라. "
            f"TINYINT, SMALLINT, MEDIUMINT 대신에 INT를 사용하라."
        )

        response = llm_generate(prompt, temperature=0.0)
        typ = response.strip().replace("\n", "").replace(" ", "")

        if typ == "DATE(`%Y%m%d`)":
            typ = "DATE(%Y%m%d)"

        ret = resolve_token_type(typ)

        if callback:
            callback({"type": "log", "message": f"  컬럼 '{columns[i]}' → {ret or 'TEXT'}"})

        if ret is None:
            results.append("TEXT")
            fields += columns[i] + ','
        elif "DATE" in ret or "date" in ret or "TIME" in ret or "time" in ret:
            tm  = ret.split("(")[0].strip()
            if "(" not in ret:
                fmt = ""
            else:
                fmt = ret.split("(")[1].strip()
                fmt = "'" + fmt.replace(")", "'")
            if not "%d" in fmt or not "%Y" in fmt or not "%m" in fmt:
                results.append("VARCHAR(10)")
                fields += columns[i] + ','
            else:
                results.append(tm)
                temp_var = var_name + str(i)
                fields  += temp_var + ','
                set_stm += f" {columns[i]} = STR_TO_DATE({temp_var}, {fmt}),"
        else:
            fields  += columns[i] + ','
            results.append(ret)

    trailing_set = set_stm[:-1] if set_stm else ""
    return results, fields[:-1] + ")\n", trailing_set


def process_directory(directory, callback=None):
    """
    directory 내 CSV 파일을 MySQL에 로드한다.
    callback(event_dict): 진행 이벤트 콜백 (optional)
      이벤트 타입:
        {"type": "log",          "message": str}
        {"type": "table_preview","filename": str, "columns": list, "rows": list}
    """
    db_name = os.path.basename(os.path.normpath(directory))

    def log(msg):
        write_log(db_name, msg)

    try:
        conn   = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()

        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{db_name}` "
            f"DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
        )
        cursor.execute(f"USE `{db_name}`")
        cursor.execute("SET SESSION sql_mode = 'STRICT_ALL_TABLES'")

        csv_files = [f for f in os.listdir(directory) if f.endswith(".csv")]
        log(f"MySQL 로딩: 총 {len(csv_files)}개 파일")

        for filename in csv_files:
            file_path  = os.path.abspath(os.path.join(directory, filename))
            table_name = os.path.splitext(filename)[0]
            log(f"[{table_name}] 타입 분석 중...")

            df_sample    = pd.read_csv(file_path, nrows=20, index_col=False, encoding='utf-8-sig') 
            column_names = df_sample.columns.tolist()

            sql_types, fields, set_stm = get_optimal_types(df_sample, callback=callback)
            log(f"[{table_name}] 타입: {sql_types}")

            if len(sql_types) != len(column_names):
                sql_types = ["TEXT"] * len(column_names)

            col_definitions = [f"`{name}` {dtype}" for name, dtype in zip(column_names, sql_types)]
            create_query    = (
                f"CREATE TABLE IF NOT EXISTS `{table_name}` "
                f"({', '.join(col_definitions)});"
            )

            cursor.execute(f"DROP TABLE IF EXISTS `{table_name}`")
            cursor.execute(create_query)

            formatted_path = file_path.replace('\\', '/')
            set_clause     = f"\nSET {set_stm}" if set_stm.strip() else ""
            load_query     = (
                f"LOAD DATA INFILE '{formatted_path}'\n"
                f"INTO TABLE `{table_name}`\n"
                f"FIELDS TERMINATED BY ','\n"
                f"ENCLOSED BY '\"'\n"
                f"LINES TERMINATED BY '\\n'\n"
                f"IGNORE 1 ROWS\n"
                f"{fields}"
                f"{set_clause};"
            )

            attempt = 1
            while True:
                try:
                    log(f"[{table_name}] 데이터 로딩 {attempt}차 시도...")
                    cursor.execute(load_query)
                    conn.commit()
                    log(f"[{table_name}] ✅ 로딩 성공!")
                    break

                except mysql.connector.Error as err:
                    if err.errno == 1406:
                        match = re.search(r"column '(.+?)'", str(err))
                        if match:
                            col_name     = match.group(1)
                            cursor.execute(f"""
                                SELECT CHARACTER_MAXIMUM_LENGTH
                                FROM information_schema.COLUMNS
                                WHERE TABLE_NAME = '{table_name}'
                                AND COLUMN_NAME = '{col_name}'
                            """)
                            current_size = cursor.fetchone()[0]
                            if current_size is None:
                                log(f"[{table_name}] 컬럼 크기 확인 불가")
                                break
                            new_size = current_size * 2
                            log(f"[{table_name}] 컬럼 '{col_name}' 크기 {current_size}→{new_size}")
                            cursor.execute(
                                f"ALTER TABLE `{table_name}` MODIFY `{col_name}` VARCHAR({new_size})"
                            )
                            attempt += 1
                            continue
                        else:
                            raise

                    elif err.errno in (1265, 1366):
                        match = re.search(r"column '(.+?)'", str(err))
                        if match:
                            col_name = match.group(1)
                            log(f"[{table_name}] 컬럼 '{col_name}' → VARCHAR(10)")
                            cursor.execute(
                                f"ALTER TABLE `{table_name}` MODIFY `{col_name}` VARCHAR(10)"
                            )
                            attempt += 1
                            continue
                        else:
                            raise
                    else:
                        log(f"[{table_name}] MySQL 에러: {err}")
                        raise

            # 파일 처리 완료: 상위 5줄 미리보기 전송
            try:
                preview_df = pd.read_csv(file_path, nrows=5, encoding='utf-8')
                columns    = preview_df.columns.tolist()
                rows       = [[str(v) for v in row] for row in preview_df.values.tolist()]
                if callback:
                    callback({
                        "type":     "table_preview",
                        "filename": filename,
                        "columns":  columns,
                        "rows":     rows,
                    })
            except Exception as e:
                log(f"[{table_name}] 미리보기 생성 실패: {e}")

        log("MySQL 로딩 전체 완료!")

    except Exception as e:
        log(f"오류 발생: {e}")
        raise
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()
