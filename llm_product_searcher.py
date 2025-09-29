# -*- coding: utf-8 -*-
import sqlite3
import re
import unicodedata
import json
import os
import logging
import atexit
from typing import List, Dict

from dotenv import load_dotenv
from kiwipiepy import Kiwi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# LLM이 반환할 JSON 구조를 Pydantic 스키마로 정의
class Keywords(BaseModel):
    keywords: List[str]


# 시스템 프롬프트: LLM의 역할과 규칙을 정의
SYSTEM_PROMPT = """너는 문장에서 농축수산물의 품목만 추출하는 어시스턴트다. 다음 규칙을 반드시 지켜라:
1) 품목만 추출하고, 품종/부위/가공형태/등급/브랜드는 제외한다. (예: ‘돼지고기’는 품목, ‘삼겹살’은 부위)
2) 각 품목의 유사어와 동의어를 충분히 고려하되, 의미가 중복되는 표현만 선별한다.
3) 문장에 2개 이상의 품목이 있으면 모두 추출한다."""

# 사용자 프롬프트: 실제 수행할 작업과 예시를 제공
USER_PROMPT = """다음 문장에서 농축수산물의 "품목"과 관련된 검색 키워드를 추출해줘.

문장: "{query}"

예시:
입력: "사과랑 배 가격"
출력: ["사과", "배"]

입력: "돼지고기 삼겹살 가격"
출력: ["돼지고기", "돼지", "돈육"]

입력: "소고기 안심 시세"
출력: ["한우", "소", "소고기", "쇠고기", "우육"]"""


class LLMProductSearcher:
    """자연어 쿼리를 받아 LLM으로 품목 키워드를 추출하고, FTS5로 DB에서 상품을 검색하는 클래스"""

    _SIMPLE_CLEAN_RE = re.compile(r"[^\w\s가-힣]")
    _WHITESPACE_RE = re.compile(r"\s+")

    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at: {db_path}")

        self.kiwi = Kiwi()
        self._llm = None
        self._structured_llm = None
        self._prompt = None

        # DB 연결 (프로세스 종료 시 자동 close)
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        atexit.register(self._connection.close)

        self._setup_database()
        self._ensure_indexes()

    # --- LLM 인스턴스 (Lazy Loading) ---
    @property
    def llm(self) -> ChatOpenAI | None:
        """LLM 인스턴스를 처음 사용할 때 로딩하여 메모리 효율성 증대"""
        if self._llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            model_name = os.getenv("OPENAI_MODEL", "gpt-5-nano")
            if api_key:
                self._llm = ChatOpenAI(
                    model=model_name,
                    temperature=0,
                    api_key=api_key,
                    reasoning_effort="minimal",
                )
        return self._llm

    @property
    def structured_llm(self):
        """Pydantic 스키마(Keywords)를 사용하여 구조화된 출력(JSON)을 강제하는 LLM"""
        if self._structured_llm is None and self.llm is not None:
            self._structured_llm = self.llm.with_structured_output(Keywords)
        return self._structured_llm

    @property
    def prompt(self) -> ChatPromptTemplate:
        """LLM에 전달할 프롬프트 템플릿 (시스템 역할 + 사용자 요청)"""
        if self._prompt is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", SYSTEM_PROMPT),
                    ("user", USER_PROMPT),
                ]
            )
        return self._prompt

    # --- Public Methods ---
    def get_name_code_pairs_json(self, natural_query: str) -> str:
        """자연어 쿼리를 받아 상품명과 코드 쌍을 JSON 문자열로 반환"""
        expanded_keywords = self._extract_product_names(natural_query)
        search_results = self._search_products(expanded_keywords)

        seen = set()
        pairs = []
        for r in search_results:
            key = (r["name"], r["code"])
            if key not in seen:
                pairs.append({"product_name": r["name"], "product_code": r["code"]})
                seen.add(key)

        return json.dumps(pairs, ensure_ascii=False, indent=2)

    def get_name_code_pairs(self, natural_query: str) -> List[Dict[str, str]]:
        """자연어 쿼리를 받아 상품명과 코드 쌍을 리스트로 반환"""
        return json.loads(self.get_name_code_pairs_json(natural_query))

    # --- DB 및 FTS5 설정 ---
    def _setup_database(self):
        """검색에 필요한 테이블(items_clean, items_fts)을 준비"""
        cur = self._connection.cursor()

        # 원본의 안전한 로직으로 복원
        items_clean_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='items_clean'"
        ).fetchone()
        if not items_clean_exists:
            cur.execute(
                """
                CREATE TABLE items_clean AS
                SELECT DISTINCT product_code, product_name
                FROM api_items
                WHERE product_code IS NOT NULL AND product_name IS NOT NULL
            """
            )

        try:
            cur.execute("ALTER TABLE items_clean ADD COLUMN tokenized_name TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        fts_exists = cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='items_fts'"
        ).fetchone()
        if not fts_exists:
            self._create_fts_table(cur)
        else:
            self._update_tokenized_data(cur)

        self._connection.commit()
        logging.info("Korean-optimized FTS5 database ready.")

    def _create_fts_table(self, cur: sqlite3.Cursor):
        """FTS5 가상 테이블을 생성하고 초기 데이터를 채움"""
        rows = cur.execute("SELECT rowid, product_name FROM items_clean").fetchall()
        for row in rows:
            tokenized = self._normalize_text(row["product_name"])
            cur.execute(
                "UPDATE items_clean SET tokenized_name = ? WHERE rowid = ?",
                (tokenized, row["rowid"]),
            )

        cur.execute(
            """
            CREATE VIRTUAL TABLE items_fts 
            USING fts5(product_name, tokenized_name, content='items_clean', content_rowid='rowid')
        """
        )
        cur.execute("INSERT INTO items_fts(items_fts) VALUES('rebuild')")
        logging.info("FTS5 table created and indexed.")

    def _update_tokenized_data(self, cur: sqlite3.Cursor):
        """기존 데이터 중 토큰화되지 않은 항목을 찾아 업데이트"""
        rows = cur.execute(
            "SELECT rowid, product_name FROM items_clean WHERE tokenized_name IS NULL"
        ).fetchall()
        if rows:
            for row in rows:
                tokenized = self._normalize_text(row["product_name"])
                cur.execute(
                    "UPDATE items_clean SET tokenized_name = ? WHERE rowid = ?",
                    (tokenized, row["rowid"]),
                )
            logging.info(f"Updated {len(rows)} tokenized records.")

    def _ensure_indexes(self):
        """검색 성능 향상을 위한 DB 인덱스 생성"""
        cur = self._connection.cursor()
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_clean_code ON items_clean(product_code)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_clean_name ON items_clean(product_name)"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_items_clean_tok ON items_clean(tokenized_name)"
        )
        self._connection.commit()

    def refresh_fts_index(self):
        """FTS 인덱스를 수동으로 재생성"""
        cur = self._connection.cursor()
        cur.execute("DROP TABLE IF EXISTS items_clean")
        cur.execute("DROP TABLE IF EXISTS items_fts")
        self._setup_database()
        logging.info("FTS index manually refreshed with latest data.")

    # --- 텍스트 정규화 (원본 구조로 복원) ---
    def _normalize_text_internal(self, s: str) -> str:
        if not s:
            return ""
        s = unicodedata.normalize("NFKC", s).strip()
        s = self._SIMPLE_CLEAN_RE.sub(" ", s)
        s = self._WHITESPACE_RE.sub(" ", s).strip()

        tokens = self.kiwi.tokenize(s)
        # 명사, 동사 어근 등 의미 있는 품사 위주로 남김
        kept_tokens = [
            t.form
            for t in tokens
            if not (
                t.tag.startswith("J") or t.tag.startswith("E") or t.tag.startswith("S")
            )
        ]
        return " ".join(kept_tokens)

    def _normalize_text(self, s: str) -> str:
        return self._normalize_text_internal(s)

    # --- LLM 키워드 추출 ---
    def _extract_product_names(self, query: str) -> List[str]:
        """LLM을 호출하여 쿼리에서 품목 키워드 리스트를 추출 (실패 시 정규화된 쿼리 전체를 키워드로 사용)"""
        if self.structured_llm is None:
            logging.warning("LLM unavailable. Falling back to normalized query.")
            return [self._normalize_text(query)]

        chain = self.prompt | self.structured_llm
        try:
            result: Keywords = chain.invoke({"query": query})
            keywords = result.keywords or []
        except Exception as e:
            logging.warning(f"Structured output failed: {e}; falling back.")
            keywords = []

        # LLM 출력 결과 후처리 (길이 제한, 정규화, 중복 제거)
        keywords = self._sanitize_keywords(keywords, max_items=10, max_len=12)
        keywords = [self._normalize_text(k) for k in keywords if k]
        keywords = list(dict.fromkeys(k for k in keywords if k))

        # LLM이 키워드를 반환하지 못한 경우, 사용자 쿼리 자체를 검색어로 사용
        return keywords or [self._normalize_text(query)]

    @staticmethod
    def _sanitize_keywords(xs: List[str], max_items=10, max_len=12) -> List[str]:
        """LLM이 생성한 키워드 리스트를 안전하게 정제"""
        out = []
        for x in xs:
            s = (x or "").strip()
            if s:
                out.append(s[:max_len])
            if len(out) >= max_items:
                break
        return list(dict.fromkeys(out))

    # --- FTS5 DB 검색 ---
    def _search_products(self, keywords: List[str]) -> list:
        """키워드 리스트를 사용하여 FTS5 DB에서 상품을 검색 (FTS 실패 시 LIKE 검색으로 대체)"""
        if not keywords:
            return []

        results = []
        seen_codes = set()
        cur = self._connection.cursor()

        for keyword in keywords:
            q = keyword
            if not q:
                continue

            try:
                match_rows = cur.execute(
                    """
                    SELECT t1.product_code, t1.product_name
                    FROM items_clean AS t1
                    JOIN (
                        SELECT rowid FROM items_fts WHERE tokenized_name MATCH ?
                        UNION
                        SELECT rowid FROM items_fts WHERE product_name MATCH ?
                    ) AS t2 ON t1.rowid = t2.rowid
                    ORDER BY 
                        CASE WHEN t1.product_name LIKE ? || '%' THEN 0 ELSE 1 END,
                        length(t1.product_name)
                    LIMIT 10
                    """,
                    (q, q, q),
                ).fetchall()
                search_method = "fts5_match"
            except sqlite3.OperationalError as e:
                logging.warning(f"FTS5 MATCH failed, falling back to LIKE: {e}")
                match_rows = cur.execute(
                    """
                    SELECT product_code, product_name
                    FROM items_clean
                    WHERE tokenized_name LIKE '%' || ? || '%'
                       OR product_name LIKE '%' || ? || '%'
                    ORDER BY 
                        CASE WHEN product_name LIKE ? || '%' THEN 0 ELSE 1 END,
                        length(product_name)
                    LIMIT 10
                    """,
                    (q, q, q),
                ).fetchall()
                search_method = "like_fallback"

            for row in match_rows:
                code = row["product_code"]
                if code not in seen_codes:
                    results.append(
                        {
                            "code": code,
                            "name": row["product_name"],
                            "method": search_method,
                            "matched_keyword": keyword,
                        }
                    )
                    seen_codes.add(code)
        return results


if __name__ == "__main__":
    DB_PATH = os.getenv("DB_PATH", "kamis_api_list.db")
    QUERY = os.getenv("QUERY", "마늘 가격을 알려줘.")

    try:
        searcher = LLMProductSearcher(DB_PATH)
        print(f"Original Query: {QUERY}\n")
        json_output = searcher.get_name_code_pairs_json(QUERY)
        print("--- Result (JSON) ---")
        print(json_output)
    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
