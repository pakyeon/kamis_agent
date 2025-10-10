# -*- coding: utf-8 -*-
import sqlite3
import re
import unicodedata
import json
import os
import logging
import atexit
from typing import List, Dict, Set, Optional
from collections import defaultdict

from dotenv import load_dotenv
from kiwipiepy import Kiwi
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class HierarchicalKeywords(BaseModel):
    categories: List[str] = []
    products: List[str] = []
    kinds: List[str] = []
    grades: List[str] = []


SYSTEM_PROMPT = """너는 농축수산물 유통 정보 시스템의 검색 어시스턴트다. 
사용자의 자연어 질문을 분석하여 다음 계층별로 키워드를 추출해야 한다:

1. 부류(category): 큰 분류 (예: 식량작물, 채소류, 특용작물, 과일류, 축산물, 수산물)
2. 품목(product): 구체적인 상품명 (예: 쌀, 배추, 사과, 돼지, 고등어)
3. 품종(kind): 품목의 세부 종류 (예: 캠벨얼리, 월동, 후지, 삼겹살, 안심)
4. 등급(grade): 품질 등급 (예: 상품, 중품, 하품, 특大, 大, 中, 小)

중요한 규칙:
- 각 계층의 유사어/동의어를 포함할 수 있다.
- 없는 계층은 빈 리스트로 반환한다.
- 품종과 등급이 명시되지 않으면 추측하지 말고 빈 리스트로 둔다."""

USER_PROMPT = """다음 사용자 질문을 분석하여 계층별 키워드를 추출해줘.

질문: "{query}"

입력: "돼지고기 삼겹살 1등급 가격 알려줘"
출력: {{"categories": ["축산물"], "products": ["돼지고기", "돼지"], "kinds": ["삼겹살"], "grades": ["1등급"]}}

입력: "한우 등심 1++등급은 얼마야?"
출력: {{"categories": ["축산물"], "products": ["소고기", "소"], "kinds": ["등심"], "grades": ["1++등급"]}}

입력: "후지 사과 상품 가격"
출력: {{"categories": ["과일류"], "products": ["사과"], "kinds": ["후지"], "grades": ["상품"]}}"""


class LLMHierarchicalSearcher:
    """계층적 구조를 고려한 농축수산물 검색 클래스"""

    _SIMPLE_CLEAN_RE = re.compile(r"[^\w\s가-힣]")
    _WHITESPACE_RE = re.compile(r"\s+")
    _DIGIT_RE = re.compile(r"\d+")
    LIVESTOCK_CATEGORY_CODE = "500"

    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at: {db_path}")

        self.kiwi = Kiwi()
        self._llm = self._structured_llm = self._prompt = None
        self._connection = sqlite3.connect(db_path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        atexit.register(self._connection.close)

        self._setup_database()
        self._ensure_indexes()

    @property
    def llm(self) -> Optional[ChatOpenAI]:
        if self._llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                self._llm = ChatOpenAI(
                    model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
                    temperature=0,
                    reasoning_effort="minimal",
                    api_key=api_key,
                )
        return self._llm

    @property
    def structured_llm(self):
        if self._structured_llm is None and self.llm:
            self._structured_llm = self.llm.with_structured_output(HierarchicalKeywords)
        return self._structured_llm

    @property
    def prompt(self) -> ChatPromptTemplate:
        if self._prompt is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
            )
        return self._prompt

    def search_hierarchical(self, natural_query: str) -> List[Dict]:
        """자연어 쿼리로 계층적 검색 수행 - 각 조합을 개별 항목으로 반환"""
        keywords = self._extract_hierarchical_keywords(natural_query)
        logging.info(f"추출된 키워드: {keywords}")
        raw_results = self._search_with_hierarchy(keywords)
        return self._organize_hierarchy(raw_results)

    def get_full_hierarchy_json(self, natural_query: str) -> str:
        """검색 결과를 JSON 문자열로 반환"""
        return json.dumps(
            self.search_hierarchical(natural_query), ensure_ascii=False, indent=2
        )

    def _setup_database(self):
        """기존 FTS5 설정 유지"""
        cur = self._connection.cursor()

        if not cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='items_clean'"
        ).fetchone():
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
            pass

        if not cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='items_fts'"
        ).fetchone():
            self._create_fts_table(cur)
        else:
            self._update_tokenized_data(cur)

        self._connection.commit()
        logging.info("Database ready for hierarchical search.")

    def _create_fts_table(self, cur: sqlite3.Cursor):
        for row in cur.execute(
            "SELECT rowid, product_name FROM items_clean"
        ).fetchall():
            cur.execute(
                "UPDATE items_clean SET tokenized_name = ? WHERE rowid = ?",
                (self._normalize_text(row["product_name"]), row["rowid"]),
            )

        cur.execute(
            """
            CREATE VIRTUAL TABLE items_fts 
            USING fts5(product_name, tokenized_name, content='items_clean', content_rowid='rowid')
        """
        )
        cur.execute("INSERT INTO items_fts(items_fts) VALUES('rebuild')")

    def _update_tokenized_data(self, cur: sqlite3.Cursor):
        for row in cur.execute(
            "SELECT rowid, product_name FROM items_clean WHERE tokenized_name IS NULL"
        ).fetchall():
            cur.execute(
                "UPDATE items_clean SET tokenized_name = ? WHERE rowid = ?",
                (self._normalize_text(row["product_name"]), row["rowid"]),
            )

    def _ensure_indexes(self):
        """계층적 검색을 위한 인덱스 추가 (축산물 등급 포함)"""
        cur = self._connection.cursor()
        for idx_name, col_name in [
            ("idx_category_code", "category_code"),
            ("idx_product_code", "product_code"),
            ("idx_kind_code", "kind_code"),
            ("idx_livestock_kind", "livestock_kind_code"),
            ("idx_productrank", "productrank_code"),
            ("idx_graderank", "graderank_code"),
            ("idx_category_name", "category_name"),
            ("idx_kind_name", "kind_name"),
            ("idx_rank_name", "rank_name"),
            ("idx_period_product_list", "p_periodProductList"),
            ("idx_period_product_name", "p_periodProductName"),
        ]:
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON api_items({col_name})"
            )
        self._connection.commit()

    def _normalize_text(self, s: str) -> str:
        if not s:
            return ""
        s = unicodedata.normalize("NFKC", s).strip()
        s = self._SIMPLE_CLEAN_RE.sub(" ", s)
        s = self._WHITESPACE_RE.sub(" ", s).strip()

        tokens = [
            t.form
            for t in self.kiwi.tokenize(s)
            if not (
                t.tag.startswith("J") or t.tag.startswith("E") or t.tag.startswith("S")
            )
        ]
        return " ".join(tokens)

    def _extract_hierarchical_keywords(self, query: str) -> HierarchicalKeywords:
        """LLM으로 계층별 키워드 추출"""
        if not self.structured_llm:
            logging.warning("LLM unavailable. Using query as product keyword.")
            return HierarchicalKeywords(products=[self._normalize_text(query)])

        try:
            result = (self.prompt | self.structured_llm).invoke({"query": query})
            return HierarchicalKeywords(
                categories=[self._normalize_text(k) for k in result.categories if k],
                products=[self._normalize_text(k) for k in result.products if k],
                kinds=[self._normalize_text(k) for k in result.kinds if k],
                grades=[self._normalize_text(k) for k in result.grades if k],
            )
        except Exception as e:
            logging.warning(f"LLM extraction failed: {e}")
            return HierarchicalKeywords(products=[self._normalize_text(query)])

    def _search_with_hierarchy(self, keywords: HierarchicalKeywords) -> List[Dict]:
        """계층 정보를 활용한 하이브리드 검색"""
        cur = self._connection.cursor()
        product_codes = set()

        # 1단계: 품목 검색
        if keywords.products:
            product_codes = self._search_by_keywords(keywords.products, "product", cur)
            logging.info(f"품목 키워드로 검색된 코드: {product_codes}")

        # 2단계: 부류로 필터링
        if keywords.categories:
            category_codes = self._search_by_keywords(
                keywords.categories, "category", cur
            )
            logging.info(f"부류 코드: {category_codes}")

            if product_codes:
                product_codes = self._filter_by_codes(
                    product_codes, category_codes, "category", cur
                )
                logging.info(f"부류 필터링 후: {product_codes}")
            else:
                product_codes = self._get_products_by_category(category_codes, cur)
                logging.info(f"부류로만 검색: {len(product_codes)}개")

        # 품목이 없으면 기타 계층으로 검색
        if not product_codes and not keywords.products:
            product_codes = self._search_by_other_hierarchy(keywords, cur)
            logging.info(f"기타 계층으로 검색: {len(product_codes)}개")

        if not product_codes:
            logging.warning("검색된 품목 코드가 없습니다.")
            return []

        # 3단계: 전체 계층 정보 조회
        results = []
        for product_code in product_codes:
            items = self._get_full_hierarchy(product_code, keywords, cur)
            results.extend(items)
            logging.info(f"품목 {product_code}: {len(items)}개 항목")

        return results

    def _search_by_keywords(
        self, keywords: List[str], search_type: str, cur: sqlite3.Cursor
    ) -> Set[str]:
        """통합 키워드 검색 (품목/부류)"""
        codes = set()

        if search_type == "product":
            code_col, name_col = "product_code", "product_name"
        else:  # category
            code_col, name_col = "category_code", "category_name"

        for keyword in keywords:
            if not keyword:
                continue

            # FTS5 검색 (품목만)
            if search_type == "product":
                try:
                    codes.update(
                        row[code_col]
                        for row in cur.execute(
                            """
                        SELECT DISTINCT t1.product_code
                        FROM items_clean AS t1
                        JOIN items_fts AS t2 ON t1.rowid = t2.rowid
                        WHERE t2.tokenized_name MATCH ? OR t2.product_name MATCH ?
                        LIMIT 50
                    """,
                            (keyword, keyword),
                        ).fetchall()
                    )
                except sqlite3.OperationalError:
                    pass

            # 직접 검색
            codes.update(
                row[code_col]
                for row in cur.execute(
                    f"""
                SELECT DISTINCT {code_col}
                FROM api_items
                WHERE {name_col} LIKE '%' || ? || '%'
                LIMIT 50
            """,
                    (keyword,),
                ).fetchall()
            )

        return codes

    def _filter_by_codes(
        self,
        product_codes: Set[str],
        filter_codes: Set[str],
        filter_type: str,
        cur: sqlite3.Cursor,
    ) -> Set[str]:
        """코드로 필터링"""
        if not product_codes or not filter_codes:
            return product_codes

        filter_col = f"{filter_type}_code"
        placeholders = ",".join("?" * len(product_codes))
        filter_placeholders = ",".join("?" * len(filter_codes))

        return {
            row["product_code"]
            for row in cur.execute(
                f"""
            SELECT DISTINCT product_code
            FROM api_items
            WHERE product_code IN ({placeholders})
              AND {filter_col} IN ({filter_placeholders})
        """,
                list(product_codes) + list(filter_codes),
            ).fetchall()
        }

    def _get_products_by_category(
        self, category_codes: Set[str], cur: sqlite3.Cursor
    ) -> Set[str]:
        """부류 코드로 품목 코드 조회"""
        if not category_codes:
            return set()

        placeholders = ",".join("?" * len(category_codes))
        return {
            row["product_code"]
            for row in cur.execute(
                f"""
            SELECT DISTINCT product_code
            FROM api_items
            WHERE category_code IN ({placeholders})
            LIMIT 100
        """,
                list(category_codes),
            ).fetchall()
        }

    def _search_by_other_hierarchy(
        self, keywords: HierarchicalKeywords, cur: sqlite3.Cursor
    ) -> Set[str]:
        """품목 외 계층(품종/등급)으로 검색"""
        conditions, params = [], []

        if keywords.kinds:
            conditions.append(
                "("
                + " OR ".join(["kind_name LIKE '%' || ? || '%'"] * len(keywords.kinds))
                + ")"
            )
            params.extend(keywords.kinds)

        if keywords.grades:
            conditions.append(
                "("
                + " OR ".join(["rank_name LIKE '%' || ? || '%'"] * len(keywords.grades))
                + ")"
            )
            params.extend(keywords.grades)

        if not conditions:
            return set()

        return {
            row["product_code"]
            for row in cur.execute(
                f"""
            SELECT DISTINCT product_code
            FROM api_items
            WHERE {" OR ".join(conditions)}
            LIMIT 100
        """,
                params,
            ).fetchall()
        }

    def _get_full_hierarchy(
        self, product_code: str, keywords: HierarchicalKeywords, cur: sqlite3.Cursor
    ) -> List[Dict]:
        """품목 코드로 전체 계층 정보 조회"""
        is_livestock = cur.execute(
            """
            SELECT category_code FROM api_items
            WHERE product_code = ?
            LIMIT 1
        """,
            (product_code,),
        ).fetchone()

        is_livestock = (
            is_livestock
            and is_livestock["category_code"] == self.LIVESTOCK_CATEGORY_CODE
        )

        return (
            self._get_livestock_hierarchy(product_code, keywords, cur)
            if is_livestock
            else self._get_general_hierarchy(product_code, keywords, cur)
        )

    def _build_hierarchy_query(
        self, product_code: str, keywords: HierarchicalKeywords, is_livestock: bool
    ) -> tuple:
        """계층 쿼리 빌드 (통합)"""
        if is_livestock:
            base_query = """
                SELECT DISTINCT
                    category_code, category_name, product_code, product_name,
                    livestock_kind_code, kind_name,
                    p_periodProductList, p_periodProductName
                FROM api_items
                WHERE product_code = ? AND category_code = ?
            """
            params = [product_code, self.LIVESTOCK_CATEGORY_CODE]
        else:
            base_query = """
                SELECT DISTINCT
                    category_code, category_name, product_code, product_name,
                    kind_code, kind_name,
                    productrank_code, graderank_code, rank_name
                FROM api_items
                WHERE product_code = ? AND category_code != ?
            """
            params = [product_code, self.LIVESTOCK_CATEGORY_CODE]

        # 품종 필터링
        if keywords.kinds:
            base_query += f" AND ({' OR '.join(['kind_name LIKE ? || ? || ?'] * len(keywords.kinds))})"
            params.extend(["%", k, "%"] for k in keywords.kinds)
            params = [
                item
                for sublist in params
                for item in (sublist if isinstance(sublist, list) else [sublist])
            ]

        # 등급 필터링
        if keywords.grades:
            if is_livestock:
                grade_patterns = []
                for grade in keywords.grades:
                    grade_patterns.extend(
                        [
                            "p_periodProductName LIKE '%' || ? || '%'",
                            "p_periodProductList LIKE '%' || ? || '%'",
                        ]
                    )
                    params.extend([grade, grade])

                    # 숫자 추출
                    for digit in self._DIGIT_RE.findall(grade):
                        grade_patterns.append("p_periodProductList = ?")
                        params.append(digit)

                if grade_patterns:
                    base_query += f" AND ({' OR '.join(grade_patterns)})"
            else:
                base_query += f" AND ({' OR '.join(['rank_name LIKE ? || ? || ?', 'graderank_code LIKE ? || ? || ?'] * len(keywords.grades))})"
                for grade in keywords.grades:
                    params.extend(["%", grade, "%", "%", grade, "%"])

        return base_query, params

    def _get_general_hierarchy(
        self, product_code: str, keywords: HierarchicalKeywords, cur: sqlite3.Cursor
    ) -> List[Dict]:
        """농수산물 계층 정보 조회"""
        query, params = self._build_hierarchy_query(product_code, keywords, False)
        rows = cur.execute(query, params).fetchall()
        logging.info(f"농수산물 검색 결과: {len(rows)}행")

        return [
            {
                "category_code": row["category_code"],
                "category_name": row["category_name"],
                "product_code": row["product_code"],
                "product_name": row["product_name"],
                "item_type": "general",
                **(
                    {"kind_code": row["kind_code"], "kind_name": row["kind_name"]}
                    if row["kind_code"]
                    else {}
                ),
                **(
                    {
                        "grade": {
                            "productrank_code": row["productrank_code"],
                            "graderank_code": row["graderank_code"],
                            "rank_name": row["rank_name"],
                        }
                    }
                    if row["productrank_code"]
                    else {}
                ),
            }
            for row in rows
        ]

    def _get_livestock_hierarchy(
        self, product_code: str, keywords: HierarchicalKeywords, cur: sqlite3.Cursor
    ) -> List[Dict]:
        """축산물 계층 정보 조회"""
        query, params = self._build_hierarchy_query(product_code, keywords, True)
        rows = cur.execute(query, params).fetchall()
        logging.info(f"축산물 검색 결과: {len(rows)}행")

        return [
            {
                "category_code": row["category_code"],
                "category_name": row["category_name"],
                "product_code": row["product_code"],
                "product_name": row["product_name"],
                "item_type": "livestock",
                **(
                    {
                        "kind_code": row["livestock_kind_code"],
                        "kind_name": row["kind_name"],
                    }
                    if row["livestock_kind_code"]
                    else {}
                ),
                **(
                    {
                        "grade": {
                            "period_product_list": row["p_periodProductList"],
                            "period_product_name": row["p_periodProductName"],
                        }
                    }
                    if row["p_periodProductList"]
                    else {}
                ),
            }
            for row in rows
        ]

    def _organize_hierarchy(self, raw_results: List[Dict]) -> List[Dict]:
        """검색 결과를 계층 구조로 재구성"""
        results = []

        for item in raw_results:
            result_item = {
                "category": {
                    "category_code": item["category_code"],
                    "category_name": item["category_name"],
                },
                "product": {
                    "product_code": item["product_code"],
                    "product_name": item["product_name"],
                },
            }

            # 품종 정보
            if "kind_code" in item and item["kind_code"]:
                result_item["kind"] = {
                    "kind_code": item["kind_code"],
                    "kind_name": item["kind_name"],
                }

            # 등급 정보
            if "grade" in item:
                is_livestock = item["item_type"] == "livestock"
                if is_livestock:
                    result_item["grade"] = {
                        "code": item["grade"]["period_product_list"],
                        "name": item["grade"]["period_product_name"],
                    }
                else:
                    result_item["grade"] = {
                        "productrank_code": item["grade"]["productrank_code"],
                        "graderank_code": item["grade"]["graderank_code"],
                        "name": item["grade"]["rank_name"],
                    }

            results.append(result_item)

        return results


if __name__ == "__main__":
    DB_PATH = os.getenv("DB_PATH", "kamis_api_list.db")
    QUERY = os.getenv("QUERY", "최근 고추의 가격을 알려줘.")

    try:
        searcher = LLMHierarchicalSearcher(DB_PATH)
        print(f"Query: {QUERY}\n")
        print("=== 계층적 검색 결과 ===")
        print(searcher.get_full_hierarchy_json(QUERY))
    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
