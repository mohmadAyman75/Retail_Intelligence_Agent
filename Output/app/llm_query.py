from __future__ import annotations

from difflib import get_close_matches
from pathlib import Path
import json
import re
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import duckdb
import pandas as pd
import sqlglot
from sqlglot import exp


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "Output" / "database" / "retail_intelligence.duckdb"

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "qwen2.5-coder:7b"
# The first request can include a cold model load; token budgets below still
# stay small so warm requests remain fast.
OLLAMA_TIMEOUT_SECONDS = 180
OLLAMA_KEEP_ALIVE = "10m"

# Small generation budgets keep the local 7B model responsive.
SQL_MAX_TOKENS = 180
SQL_REPAIR_MAX_TOKENS = 180
CHART_MAX_TOKENS = 80
ANSWER_MAX_TOKENS = 240
MAX_QUERY_ROWS = 200
MAX_RESULT_ROWS_FOR_LLM = 50
MAX_RETRY_ATTEMPTS = 3


TABLE_DESCRIPTIONS = {
    "zone_traffic": "One row per camera, 60-second window, and monitored zone.",
    "queue_history": "Observed and predicted camera-local queue size per time window.",
    "agent_actions": "Operational alerts and recommended actions generated per camera.",
}

COLUMN_DESCRIPTIONS = {
    "store_id": "store identifier such as place_02",
    "camera_id": "camera identifier; metrics are scoped to this camera",
    "window_start_sec": "window start in seconds from the recording start",
    "window_start": "human-readable duration from the recording start",
    "zone_id": "internal zone identifier",
    "zone_label_ar": "Arabic display name of the zone",
    "zone_kind": "zone category such as entrance, sales, queue, or outside",
    "customer_count": "distinct camera-local tracks in that zone and window; not a cross-camera person count",
    "observations": "number of tracking observations in that zone and window",
    "z_score": "standardized crowd level for anomaly detection",
    "is_anomaly": "whether the crowd reading is anomalous",
    "queue_length": "observed number of customers in the queue",
    "predicted_queue_next_window": "predicted queue size in the next window",
    "created_at": "human-readable duration when the action was created",
    "action_type": "operational action category",
    "severity": "alert severity such as low, medium, or high",
    "message_ar": "Arabic operational alert text",
}


def load_allowed_schema() -> dict[str, dict[str, str]]:
    """Load the live main-schema tables and columns from DuckDB."""
    if not DB_PATH.exists():
        raise RuntimeError(f"DuckDB database not found: {DB_PATH}")

    connection = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        rows = connection.execute(
            """
            SELECT table_name, column_name, data_type
            FROM information_schema.columns
            WHERE table_schema = 'main'
            ORDER BY table_name, ordinal_position
            """
        ).fetchall()
    finally:
        connection.close()

    schema: dict[str, dict[str, str]] = {}
    for table_name, column_name, data_type in rows:
        schema.setdefault(str(table_name).lower(), {})[
            str(column_name).lower()
        ] = str(data_type).upper()
    if not schema:
        raise RuntimeError("DuckDB does not contain any tables in the main schema.")
    return schema


def build_sql_system_prompt(schema: dict[str, dict[str, str]]) -> str:
    """Build the SQL prompt from the live schema, not a handwritten whitelist."""
    schema_lines = []
    for table_name, columns in schema.items():
        description = TABLE_DESCRIPTIONS.get(
            table_name,
            "Dynamically discovered DuckDB table.",
        )
        schema_lines.append(f"Table {table_name}\nDescription: {description}")
        for column_name, data_type in columns.items():
            column_description = COLUMN_DESCRIPTIONS.get(
                column_name,
                "Dynamically discovered column.",
            )
            schema_lines.append(
                f"- {column_name} {data_type}: {column_description}"
            )

    return (
        "You are a DuckDB SQL generator for a retail CCTV analytics dashboard.\n"
        "Generate exactly one read-only SELECT statement and return SQL only.\n"
        "Do not use Markdown fences, comments, CTEs, UNION, subqueries, SELECT *,\n"
        "database-qualified names, external files, table functions, or unsupported\n"
        "tables and columns. Prefer explicit columns and add LIMIT 200 for detail rows.\n"
        "Use window_start_sec for chronological ordering when it exists.\n"
        "The time fields window_start and created_at are human-readable duration strings,\n"
        "not SQL TIMESTAMP values.\n\n"
        "Database schema loaded at runtime:\n\n"
        + "\n\n".join(schema_lines)
    )


ALLOWED_SQL_FUNCTIONS = frozenset(
    {
        "abs",
        "avg",
        "cast",
        "coalesce",
        "count",
        "lower",
        "max",
        "min",
        "round",
        "sum",
        "upper",
    }
)
ALLOWED_CHART_TYPES = frozenset({"none", "bar", "line", "scatter"})


FEW_SHOT_MESSAGES = [
    {
        "role": "user",
        "content": "Arabic question: متى وأين كانت أعلى زحمة؟",
    },
    {
        "role": "assistant",
        "content": (
            "SELECT store_id, window_start, zone_label_ar, customer_count "
            "FROM zone_traffic "
            "ORDER BY customer_count DESC "
            "LIMIT 1"
        ),
    },
    {
        "role": "user",
        "content": "Arabic question: ما آخر طول للطابور وما التوقع للفترة القادمة؟",
    },
    {
        "role": "assistant",
        "content": (
            "SELECT store_id, window_start, zone_label_ar, queue_length, "
            "predicted_queue_next_window "
            "FROM queue_history "
            "ORDER BY window_start_sec DESC "
            "LIMIT 1"
        ),
    },
    {
        "role": "user",
        "content": "Arabic question: كم عدد قراءات الزوار عند المدخل لكل مكان؟",
    },
    {
        "role": "assistant",
        "content": (
            "SELECT store_id, SUM(customer_count) AS entrance_visits "
            "FROM zone_traffic "
            "WHERE zone_kind = 'entrance' "
            "GROUP BY store_id "
            "ORDER BY entrance_visits DESC"
        ),
    },
]


def call_ollama(
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    json_mode: bool = False,
) -> str:
    """Call the single local Ollama model with a strict output budget."""
    payload: dict[str, Any] = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,  # Ollama equivalent of max_tokens.
        },
    }
    if json_mode:
        payload["format"] = "json"

    http_request = urllib_request.Request(
        OLLAMA_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib_request.urlopen(
            http_request,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        ) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib_error.URLError as error:
        raise RuntimeError(
            "تعذر الاتصال بـ Ollama. شغّل `ollama serve` وتأكد من تنزيل "
            f"`{OLLAMA_MODEL}`."
        ) from error
    except (TimeoutError, json.JSONDecodeError) as error:
        raise RuntimeError("انتهت مهلة Ollama أو عاد برد غير صالح.") from error

    content = body.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Ollama أعاد استجابة فارغة.")
    return content.strip()


def clean_generated_sql(model_output: str) -> str:
    """Remove common Markdown wrappers without changing SQL semantics."""
    cleaned = model_output.strip()
    cleaned = re.sub(r"^```(?:sql)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    match = re.search(r"\bSELECT\b", cleaned, flags=re.IGNORECASE)
    if match:
        cleaned = cleaned[match.start():]
    return cleaned.strip().rstrip(";").strip()


def generate_sql(user_question: str) -> str:
    """Generate one DuckDB SELECT statement from an Arabic user question."""
    schema = load_allowed_schema()
    messages = [
        {"role": "system", "content": build_sql_system_prompt(schema)},
        *FEW_SHOT_MESSAGES,
        {
            "role": "user",
            "content": (
                "Arabic user question:\n"
                f"{user_question}\n\n"
                "Return one DuckDB SELECT statement only."
            ),
        },
    ]
    return clean_generated_sql(
        call_ollama(messages, max_tokens=SQL_MAX_TOKENS)
    )


def _forbidden_expression_types() -> tuple[type[exp.Expression], ...]:
    names = (
        "Alter",
        "Command",
        "Copy",
        "Create",
        "Delete",
        "Drop",
        "Insert",
        "Into",
        "Merge",
        "Pragma",
        "Transaction",
        "Update",
        "Use",
    )
    return tuple(
        expression_type
        for name in names
        if isinstance(
            expression_type := getattr(exp, name, None),
            type,
        )
    )


def _unknown_column_error(
    column_name: str,
    available_columns: set[str],
    *,
    table_name: str | None = None,
) -> str:
    """Create a repair-ready error with an exact typo suggestion."""
    suggestion = get_close_matches(
        column_name,
        sorted(available_columns),
        n=1,
        cutoff=0.5,
    )
    location = f" in table `{table_name}`" if table_name else ""
    suggestion_text = (
        f" Did you mean `{suggestion[0]}`?" if suggestion else ""
    )
    available_text = ", ".join(f"`{name}`" for name in sorted(available_columns))
    return (
        f"Unknown column `{column_name}`{location}."
        f"{suggestion_text} Available columns{location}: {available_text}."
    )


def get_sql_validation_error(sql: str) -> str | None:
    """Return a clear validation error, or None when the SQL is safe."""
    if not isinstance(sql, str) or not sql.strip():
        return "الاستعلام فارغ."

    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as error:
        return f"صيغة SQL غير صالحة: {error}"

    if len(statements) != 1:
        return "مسموح باستعلام SELECT واحد فقط."

    statement = statements[0]
    if not isinstance(statement, exp.Select):
        return "مسموح بتنفيذ SELECT فقط."

    forbidden_types = _forbidden_expression_types()
    if forbidden_types and any(
        isinstance(node, forbidden_types)
        for node in statement.walk()
    ):
        return "الاستعلام يحتوي على عملية غير مسموحة؛ SELECT فقط هو المسموح."

    if any(True for _ in statement.find_all(exp.CTE)):
        return "الـ CTE غير مسموح في هذه النسخة؛ استخدم SELECT مباشر."

    try:
        allowed_schema = load_allowed_schema()
    except RuntimeError as error:
        return str(error)
    allowed_tables = frozenset(allowed_schema)
    allowed_columns = frozenset(
        column_name
        for columns in allowed_schema.values()
        for column_name in columns
    )

    table_aliases: dict[str, str] = {}
    referenced_tables: set[str] = set()
    for table in statement.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            return "Table functions والمصادر الخارجية غير مسموحة."
        table_name = table.name.lower()
        if table_name not in allowed_tables:
            available_tables = ", ".join(f"`{name}`" for name in sorted(allowed_tables))
            return (
                f"Unknown table `{table_name}`. "
                f"Available tables: {available_tables}."
            )
        if table.catalog or (table.db and table.db.lower() != "main"):
            return "أسماء قواعد البيانات أو schemas الخارجية غير مسموحة."
        referenced_tables.add(table_name)
        table_aliases[table_name] = table_name
        if table.alias:
            table_aliases[table.alias.lower()] = table_name

    if not referenced_tables:
        return "يجب أن يقرأ الاستعلام من جدول مسموح."

    for star in statement.find_all(exp.Star):
        if not isinstance(star.parent, exp.Count):
            return "SELECT * غير مسموح؛ اذكر الأعمدة المطلوبة صراحة."

    output_aliases = {
        item.alias.lower()
        for item in statement.expressions
        if item.alias
    }
    for column in statement.find_all(exp.Column):
        if column.is_star:
            continue
        column_name = column.name.lower()
        if not column.table and column_name in output_aliases:
            continue

        qualifier = column.table.lower()
        if qualifier:
            table_name = table_aliases.get(qualifier)
            if table_name is None:
                return f"اسم الجدول أو alias `{qualifier}` غير مسموح."
            if column_name not in allowed_schema[table_name]:
                return _unknown_column_error(
                    column_name,
                    set(allowed_schema[table_name]),
                    table_name=table_name,
                )
            continue

        if column_name not in allowed_columns:
            candidate_columns = set().union(
                *(set(allowed_schema[table_name]) for table_name in referenced_tables)
            )
            single_table = next(iter(referenced_tables)) if len(referenced_tables) == 1 else None
            return _unknown_column_error(
                column_name,
                candidate_columns,
                table_name=single_table,
            )
        if not any(
            column_name in allowed_schema[table_name]
            for table_name in referenced_tables
        ):
            candidate_columns = set().union(
                *(set(allowed_schema[table_name]) for table_name in referenced_tables)
            )
            single_table = next(iter(referenced_tables)) if len(referenced_tables) == 1 else None
            return _unknown_column_error(
                column_name,
                candidate_columns,
                table_name=single_table,
            )

    for function in statement.find_all(exp.Func):
        function_name = function.sql_name().lower()
        if function_name not in ALLOWED_SQL_FUNCTIONS:
            return f"الدالة SQL `{function_name}` غير مسموحة."

    return None


def validate_sql(sql: str) -> bool:
    """Return True only for one whitelisted, read-only SELECT statement."""
    return get_sql_validation_error(sql) is None


def execute_sql(sql: str) -> pd.DataFrame:
    """Execute validated SQL through a read-only DuckDB connection."""
    validation_error = get_sql_validation_error(sql)
    if validation_error:
        raise ValueError(validation_error)
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DuckDB database not found: {DB_PATH}")

    bounded_sql = (
        "SELECT * FROM ("
        f"{sql.rstrip(';')}"
        f") AS validated_query LIMIT {MAX_QUERY_ROWS}"
    )
    connection = duckdb.connect(
        str(DB_PATH),
        read_only=True,
        config={"enable_external_access": "false"},
    )
    try:
        return connection.execute(bounded_sql).fetchdf()
    finally:
        connection.close()


def repair_sql(
    user_question: str,
    failed_sql: str,
    error_message: str,
) -> str:
    """Ask the same model for one correction using the exact failure reason."""
    schema = load_allowed_schema()
    messages = [
        {"role": "system", "content": build_sql_system_prompt(schema)},
        {
            "role": "user",
            "content": (
                "Correct the failed SQL query below.\n\n"
                "Original Arabic user question:\n"
                f"{user_question}\n\n"
                "Failed SQL:\n"
                f"{failed_sql}\n\n"
                "DuckDB error or empty-result reason:\n"
                f"{error_message}\n\n"
                "Return one corrected DuckDB SELECT statement only."
            ),
        },
    ]
    return clean_generated_sql(
        call_ollama(messages, max_tokens=SQL_REPAIR_MAX_TOKENS)
    )


def dataframe_to_json(
    query_result: pd.DataFrame,
    *,
    max_rows: int = MAX_RESULT_ROWS_FOR_LLM,
) -> str:
    """Serialize a bounded set of real query rows for an LLM prompt."""
    return query_result.head(max_rows).to_json(
        orient="records",
        force_ascii=False,
        date_format="iso",
    )


def generate_chart_decision(
    user_question: str,
    query_result: pd.DataFrame,
) -> dict:
    """Return a validated chart decision; the model never writes chart code."""
    fallback = {"chart_type": "none", "x": None, "y": None}
    if query_result.empty or len(query_result.columns) < 2:
        return fallback

    columns = [str(column) for column in query_result.columns]
    messages = [
        {
            "role": "system",
            "content": (
                "You choose a visualization for a DuckDB query result. "
                "Return JSON only with exactly these keys: chart_type, x, y. "
                "chart_type must be one of: none, bar, line, scatter. "
                "x and y must be exact column names from the provided list, "
                "or null when chart_type is none. Choose none for a single "
                "fact, a text answer, or unsuitable data. Never write code."
            ),
        },
        {
            "role": "user",
            "content": (
                "Arabic user question:\n"
                f"{user_question}\n\n"
                f"Available columns: {json.dumps(columns)}\n"
                "Actual query result rows:\n"
                f"{dataframe_to_json(query_result)}"
            ),
        },
    ]

    try:
        raw_decision = call_ollama(
            messages,
            max_tokens=CHART_MAX_TOKENS,
            json_mode=True,
        )
        decision = json.loads(raw_decision)
    except (RuntimeError, json.JSONDecodeError, TypeError):
        return fallback

    if not isinstance(decision, dict):
        return fallback
    chart_type = str(decision.get("chart_type", "none")).lower()
    x_column = decision.get("x")
    y_column = decision.get("y")
    if chart_type not in ALLOWED_CHART_TYPES:
        return fallback
    if chart_type == "none":
        return fallback
    if x_column not in columns or y_column not in columns:
        return fallback
    return {"chart_type": chart_type, "x": x_column, "y": y_column}


def generate_final_answer(
    user_question: str,
    query_result: pd.DataFrame,
) -> str:
    """Write an Arabic answer grounded only in the actual SQL result."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a retail analytics assistant. Answer in clear Arabic. "
                "Use only facts and numbers explicitly present in the supplied "
                "SQL result rows. Never invent, estimate, or add a number that "
                "is not in those rows. If the rows do not support part of the "
                "question, say that the available result is insufficient. "
                "Keep the answer concise."
            ),
        },
        {
            "role": "user",
            "content": (
                "Arabic user question:\n"
                f"{user_question}\n\n"
                "Actual SQL result rows:\n"
                f"{dataframe_to_json(query_result)}"
            ),
        },
    ]
    return call_ollama(messages, max_tokens=ANSWER_MAX_TOKENS)


def run_llm_query(user_question: str) -> dict[str, Any]:
    """Run generation, validation, execution, and up to three repairs."""
    try:
        sql = generate_sql(user_question)
    except RuntimeError as error:
        return {
            "success": False,
            "error": str(error),
            "sql": None,
            "result": pd.DataFrame(),
            "chart_decision": {"chart_type": "none", "x": None, "y": None},
            "retried": False,
            "retry_count": 0,
            "attempt_count": 0,
        }

    retry_count = 0
    result = pd.DataFrame()
    last_error = ""

    while True:
        validation_error = get_sql_validation_error(sql)
        if validation_error:
            last_error = validation_error
        else:
            try:
                result = execute_sql(sql)
            except Exception as error:
                result = pd.DataFrame()
                last_error = str(error)
            else:
                if not result.empty:
                    break
                last_error = (
                    "The query executed successfully but returned zero rows."
                )

        if retry_count >= MAX_RETRY_ATTEMPTS:
            return {
                "success": False,
                "error": (
                    "تعذر إنشاء استعلام صالح بعد "
                    f"{retry_count} محاولات تصحيح. آخر سبب: {last_error}"
                ),
                "sql": sql,
                "result": result,
                "chart_decision": {"chart_type": "none", "x": None, "y": None},
                "retried": retry_count > 0,
                "retry_count": retry_count,
                "attempt_count": retry_count + 1,
            }

        retry_count += 1
        try:
            sql = repair_sql(
                user_question,
                sql,
                last_error,
            )
        except RuntimeError as error:
            return {
                "success": False,
                "error": (
                    f"فشلت محاولة تصحيح SQL رقم {retry_count}: {error}"
                ),
                "sql": sql,
                "result": pd.DataFrame(),
                "chart_decision": {"chart_type": "none", "x": None, "y": None},
                "retried": retry_count > 0,
                "retry_count": retry_count,
                "attempt_count": retry_count + 1,
            }

    chart_decision = generate_chart_decision(user_question, result)
    try:
        answer = generate_final_answer(user_question, result)
    except RuntimeError as error:
        return {
            "success": False,
            "error": f"تم تنفيذ SQL لكن تعذر صياغة الإجابة: {error}",
            "sql": sql,
            "result": result,
            "chart_decision": chart_decision,
            "retried": retry_count > 0,
            "retry_count": retry_count,
            "attempt_count": retry_count + 1,
        }

    return {
        "success": True,
        "answer": answer,
        "error": None,
        "sql": sql,
        "result": result,
        "chart_decision": chart_decision,
        "retried": retry_count > 0,
        "retry_count": retry_count,
        "attempt_count": retry_count + 1,
    }
