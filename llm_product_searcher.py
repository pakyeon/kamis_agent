# -*- coding: utf-8 -*-
"""
LLM 강화 상품 검색 (클래스 및 FTS5 적용 버전)
자연어 쿼리 → 품목 확장 키워드 추출 → FTS5 검색 → (product_name, product_code) JSON 반환

Public API:
- LLMProductSearcher(db_path).get_name_code_pairs(natural_query)
- LLMProductSearcher(db_path).get_name_code_pairs_json(natural_query)
"""

import sqlite3
import re
import unicodedata
import json
import os
import logging
from typing import List, Dict

from dotenv import load_dotenv
from kiwipiepy import Kiwi

# 외부 LLM 사용 관련
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

# .env 파일 로드
load_dotenv()

# 로깅 기본 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LLMProductSearcher:
    """LLM과 FTS5를 사용하여 상품을 검색하는 클래스"""

    def __init__(self, db_path: str):
        """
        초기화 시 DB 경로를 확인하고, Kiwi 형태소 분석기를 로드하며,
        DB 테이블 및 FTS5 인덱스를 설정합니다.
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found at: {db_path}")
        self.db_path = db_path
        self.kiwi = Kiwi()
        self._setup_database()

    def get_name_code_pairs(self, natural_query: str) -> List[Dict[str, str]]:
        """
        자연어 쿼리를 받아 (product_name, product_code) 페어 리스트를 반환합니다.
        """
        expanded_keywords = self._extract_product_names(natural_query)
        search_results = self._search_products(expanded_keywords)

        # name-code 페어 (중복 제거, 순서 유지)
        seen = set()
        name_code_pairs = []
        for r in search_results:
            key = (r["name"], r["code"])
            if key not in seen:
                name_code_pairs.append(
                    {"product_name": r["name"], "product_code": r["code"]}
                )
                seen.add(key)
        return name_code_pairs

    def get_name_code_pairs_json(self, natural_query: str) -> str:
        """
        자연어 쿼리를 받아 (product_name, product_code) 페어 리스트를 JSON 문자열로 반환합니다.
        """
        pairs = self.get_name_code_pairs(natural_query)
        return json.dumps(pairs, ensure_ascii=False, indent=2)

    def _setup_database(self):
        """
        DB에 원본 데이터 테이블과 FTS5 가상 테이블을 생성합니다.
        FTS5 테이블은 텍스트 검색 속도를 크게 향상시킵니다.
        """
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            # 1. 중복 제거된 원본 데이터 테이블 생성
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS items_clean AS
                SELECT DISTINCT product_code, product_name
                FROM api_items
                WHERE product_code IS NOT NULL AND product_name IS NOT NULL
            """
            )

            # 2. FTS5 가상 테이블 생성
            cur.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS items_fts 
                USING fts5(product_name, content=items_clean, content_rowid=rowid)
            """
            )

            # 3. FTS 인덱스 데이터 동기화
            cur.execute("INSERT INTO items_fts(items_fts) VALUES('rebuild')")
            con.commit()
        logging.info("Database and FTS5 index are ready.")

    def _normalize_text(self, s: str) -> str:
        """텍스트 정규화: NFKC + 특수문자 제거 + 공백 정리 + 조사/어미 제거"""
        if not s:
            return ""

        s = unicodedata.normalize("NFKC", s).strip()
        s = re.sub(r"[()\[\]{}·•/_-]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()

        tokens = self.kiwi.tokenize(s)
        kept_tokens = [
            t.form
            for t in tokens
            if not (
                t.tag.startswith("J") or t.tag.startswith("E") or t.tag.startswith("S")
            )
        ]

        return " ".join(kept_tokens)

    def _extract_product_names(self, query: str) -> List[str]:
        """LangChain + OpenAI를 사용하여 자연어 쿼리에서 확장된 품목명 추출"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logging.warning(
                "OPENAI_API_KEY not found. Falling back to simple normalization."
            )
            return [self._normalize_text(query)]

        llm = ChatOpenAI(
            model="gpt-5-nano",
            temperature=0,
            api_key=api_key,
            reasoning_effort="minimal",
        )
        prompt = f"""다음 문장에서 농축수산물의 "품목"과 관련된 검색 키워드를 추출해주세요.

문장: "{query}"

중요한 규칙:
1. 품목만 추출하고, 품종은 제외해주세요. (예: '돼지고기'는 품목, '삼겹살'은 품종)
2. 품목의 유사어, 동의어, 표준명을 모두 포함해주세요.
3. 농수산물, 축산물, 수산물 관련 용어만 추출해주세요.
4. 문장에 2개 이상의 품목이 있으면 모두 추출해주세요.

올바른 예시:
- "돼지고기 삼겹살 가격" → ["돼지고기", "돼지", "돈육"]
- "한우 등심 시세" → ["한우", "소", "소고기", "쇠고기", "우육"]
- "사과랑 배 가격" → ["사과", "배"]

JSON 배열로 반환: ["품목1", "품목2", ...]

답변:"""

        try:
            message = HumanMessage(content=prompt)
            response = llm.invoke([message])
            result = response.content.strip()

            products = json.loads(result)

            normalized = [self._normalize_text(p) for p in products if str(p).strip()]
            unique_products = list(
                dict.fromkeys(p for p in normalized if p)
            )  # 순서 유지하며 중복 제거
            return unique_products or [self._normalize_text(query)]

        except json.JSONDecodeError:
            # LLM이 JSON 형식이 아닌 다른 답변을 줬을 경우, 따옴표 안의 내용이라도 추출
            products = re.findall(r'"([^"]+)"', result)
            normalized = [self._normalize_text(p) for p in products if str(p).strip()]
            unique_products = list(dict.fromkeys(p for p in normalized if p))
            return unique_products or [self._normalize_text(query)]

        except Exception as e:
            # ===> 안정성 개선: 오류 로깅 추가
            logging.error(f"LLM keyword extraction failed: {e}")
            return [self._normalize_text(query)]

    def _search_products(self, keywords: List[str]) -> list:
        """FTS5 MATCH를 사용하여 DB에서 상품 정보를 검색합니다."""
        if not keywords:
            return []

        results = []
        seen_codes = set()

        with sqlite3.connect(self.db_path) as con:
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            for keyword in keywords:
                q = self._normalize_text(keyword)
                if not q:
                    continue

                # ===> 성능 개선 및 오류 수정: FTS5의 올바른 MATCH 문법 사용
                match_rows = cur.execute(
                    """
                    SELECT t1.product_code, t1.product_name
                    FROM items_clean AS t1
                    JOIN (
                        SELECT rowid FROM items_fts WHERE items_fts MATCH ?
                    ) AS t2 ON t1.rowid = t2.rowid
                    ORDER BY length(t1.product_name)
                """,
                    (q,),
                ).fetchall()

                for row in match_rows:
                    code = row["product_code"]
                    if code not in seen_codes:
                        results.append(
                            {
                                "code": code,
                                "name": row["product_name"],
                                "method": "fts5",
                                "matched_keyword": keyword,
                            }
                        )
                        seen_codes.add(code)
        return results


# -------------------------
# 모듈 단독 실행 예시 (간단 테스트)
# -------------------------
if __name__ == "__main__":
    DB_PATH = os.getenv("DB_PATH", "kamis_api_list.db")
    QUERY = os.getenv("QUERY", "오늘 돼지고기랑 배추 가격 좀 알려줘")

    try:
        # 클래스 인스턴스 생성
        searcher = LLMProductSearcher(DB_PATH)

        # 메서드 호출
        print(f"Original Query: {QUERY}\n")
        json_output = searcher.get_name_code_pairs_json(QUERY)

        # JSON 형태로 바로 출력
        print("--- Result (JSON) ---")
        print(json_output)

    except FileNotFoundError as e:
        logging.error(e)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
