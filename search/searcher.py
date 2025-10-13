# -*- coding: utf-8 -*-
"""계층적 품목 검색 엔진"""

import logging
from typing import List, Dict, Set, Optional
from pydantic import BaseModel

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from .db_manager import DatabaseManager
from .query_builder import QueryBuilder
from .text_processor import TextProcessor
from ..exceptions import DatabaseError

logger = logging.getLogger(__name__)


# LLM 추출 결과 모델
class HierarchicalKeywords(BaseModel):
    """LLM이 추출한 계층별 키워드"""

    categories: List[str] = []
    products: List[str] = []
    kinds: List[str] = []
    grades: List[str] = []


# LLM 프롬프트
SYSTEM_PROMPT = """너는 농축수산물 유통 정보 시스템의 검색 어시스턴트다.
사용자의 자연어 질문을 분석하여 다음 계층별로 키워드를 추출해야 한다:

1. 부류(category): 큰 분류 (예: 식량작물, 채소류, 특용작물, 과일류, 축산물, 수산물)
2. 품목(product): 구체적인 상품명 (예: 쌀, 배추, 사과, 돼지, 고등어)
3. 품종(kind): 품목의 세부 종류 (예: 캠벨얼리, 월동, 후지, 삼겹살, 안심)
4. 등급(grade): 품질 등급 (예: 상품, 중품, 하품, 특大, 大, 中, 小, 1등급, 1++등급)

규칙:
- 품목, 품종, 등급 계층에서는 유사어/동의어를 포함해야 한다.
- 없는 계층은 빈 리스트로 반환한다.
- 품종과 등급이 명시되지 않으면 추측하지 말고 빈 리스트로 둔다."""

USER_PROMPT = """다음 사용자 질문을 분석하여 계층별 키워드를 추출해줘.

질문: "{query}"

예시:
입력: "오늘 돼지고기 삼겹살 1등급 가격 알려줘."
출력: {{"categories": ["축산물"], "products": ["돼지고기", "돼지"], "kinds": ["삼겹살"], "grades": ["1등급"]}}

입력: "작년 한우 등심 1++등급은 얼마야?"
출력: {{"categories": ["축산물"], "products": ["소고기", "소"], "kinds": ["등심"], "grades": ["1++등급"]}}

입력: "2024년 후지 사과 상품 가격 추이를 알려줘."
출력: {{"categories": ["과일류"], "products": ["사과"], "kinds": ["후지"], "grades": ["상품"]}}"""


class HierarchicalSearcher:
    """계층적 구조를 고려한 농축수산물 검색"""

    LIVESTOCK_CATEGORY_CODE = "500"

    def __init__(
        self,
        db_path: str,
        llm: Optional[ChatOpenAI] = None,
        text_processor: Optional[TextProcessor] = None,
    ):
        """
        Args:
            db_path: SQLite DB 경로
            llm: LangChain LLM (없으면 키워드 추출 불가)
            text_processor: 텍스트 프로세서 (없으면 자동 생성)
        """
        self.db = DatabaseManager(db_path)
        self.query_builder = QueryBuilder()
        self.text_processor = text_processor or TextProcessor()

        # LLM 설정 (선택적)
        self._llm = llm
        self._structured_llm = None
        self._prompt = None

        if llm:
            self._setup_llm()

    def _setup_llm(self) -> None:
        """LLM 및 프롬프트 설정"""
        if not self._llm:
            return

        self._structured_llm = self._llm.with_structured_output(HierarchicalKeywords)
        self._prompt = ChatPromptTemplate.from_messages(
            [("system", SYSTEM_PROMPT), ("user", USER_PROMPT)]
        )

    def search(self, natural_query: str, top_k: int = 10) -> List[Dict]:
        """
        자연어 쿼리로 품목 검색

        Args:
            natural_query: 자연어 검색어 (예: "후지 사과 상품")
            top_k: 최대 반환 개수

        Returns:
            검색 결과 리스트 (계층 구조)
        """
        # 1. LLM으로 키워드 추출
        keywords = self._extract_keywords(natural_query)
        logger.info(f"추출된 키워드: {keywords}")

        # 2. 계층적 검색 수행
        raw_results = self._hierarchical_search(keywords)

        # 3. 결과 구조화
        structured = self._structure_results(raw_results)

        return structured[:top_k]

    def _extract_keywords(self, query: str) -> HierarchicalKeywords:
        """LLM으로 계층별 키워드 추출"""
        if not self._structured_llm or not self._prompt:
            # LLM 없으면 단순 정규화
            logger.warning("LLM 없음. 단순 텍스트 정규화 사용")
            normalized = self.text_processor.normalize(query)
            return HierarchicalKeywords(products=[normalized])

        try:
            chain = self._prompt | self._structured_llm
            result = chain.invoke({"query": query})

            # 텍스트 정규화 적용
            return HierarchicalKeywords(
                categories=[
                    self.text_processor.normalize(k) for k in result.categories if k
                ],
                products=[
                    self.text_processor.normalize(k) for k in result.products if k
                ],
                kinds=[self.text_processor.normalize(k) for k in result.kinds if k],
                grades=[self.text_processor.normalize(k) for k in result.grades if k],
            )
        except Exception as e:
            logger.warning(f"LLM 추출 실패: {e}. 단순 정규화 사용")
            normalized = self.text_processor.normalize(query)
            return HierarchicalKeywords(products=[normalized])

    def _hierarchical_search(self, keywords: HierarchicalKeywords) -> List[Dict]:
        """계층적 검색 수행"""
        product_codes = set()

        # 1단계: 품목 검색
        if keywords.products:
            query, params = self.query_builder.build_product_search(keywords.products)
            rows = self.db.execute(query, tuple(params))
            product_codes = {row["product_code"] for row in rows}
            logger.info(f"품목 검색: {len(product_codes)}개")

        # 2단계: 부류로 필터링
        if keywords.categories:
            query, params = self.query_builder.build_category_search(
                keywords.categories
            )
            rows = self.db.execute(query, tuple(params))
            category_codes = {row["category_code"] for row in rows}

            if product_codes:
                # 기존 품목 필터링
                query, params = self.query_builder.build_filter_by_category(
                    product_codes, category_codes
                )
                rows = self.db.execute(query, tuple(params))
                product_codes = {row["product_code"] for row in rows}
            else:
                # 부류만으로 품목 조회
                query, params = self.query_builder.build_products_by_category(
                    category_codes
                )
                rows = self.db.execute(query, tuple(params))
                product_codes = {row["product_code"] for row in rows}

            logger.info(f"부류 필터 후: {len(product_codes)}개")

        if not product_codes:
            logger.warning("검색된 품목 없음")
            return []

        # 3단계: 각 품목의 계층 정보 조회
        results = []
        for product_code in product_codes:
            items = self._get_hierarchy_info(product_code, keywords)
            results.extend(items)

        return results

    def _get_hierarchy_info(
        self, product_code: str, keywords: HierarchicalKeywords
    ) -> List[Dict]:
        """품목의 전체 계층 정보 조회"""
        # 축산물 여부 확인
        check_query = (
            "SELECT category_code FROM api_items WHERE product_code = ? LIMIT 1"
        )
        rows = self.db.execute(check_query, (product_code,))

        if not rows:
            return []

        is_livestock = rows[0]["category_code"] == self.LIVESTOCK_CATEGORY_CODE

        # 계층 정보 조회
        query, params = self.query_builder.build_hierarchy_search(
            product_code, keywords.kinds, keywords.grades, is_livestock
        )

        rows = self.db.execute(query, tuple(params))
        return [dict(row) for row in rows]

    def _structure_results(self, raw_results: List[Dict]) -> List[Dict]:
        """검색 결과를 구조화된 형태로 변환"""
        structured = []

        for item in raw_results:
            result = {
                "category": {
                    "code": item.get("category_code"),
                    "name": item.get("category_name"),
                },
                "product": {
                    "code": item.get("product_code"),
                    "name": item.get("product_name"),
                },
            }

            # 품종
            kind_code = item.get("kind_code") or item.get("livestock_kind_code")
            if kind_code:
                result["kind"] = {
                    "code": kind_code,
                    "name": item.get("kind_name"),
                }

            # 등급
            if item.get("category_code") == self.LIVESTOCK_CATEGORY_CODE:
                # 축산물
                if item.get("p_periodProductList"):
                    result["grade"] = {
                        "code": item.get("p_periodProductList"),
                        "name": item.get("p_periodProductName"),
                    }
            else:
                # 일반 품목
                if item.get("productrank_code"):
                    result["grade"] = {
                        "productrank_code": item.get("productrank_code"),
                        "graderank_code": item.get("graderank_code"),
                        "name": item.get("rank_name"),
                    }

            structured.append(result)

        return structured
