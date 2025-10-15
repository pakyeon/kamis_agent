# -*- coding: utf-8 -*-
"""계층적 품목 검색 엔진 (지역 정보 포함)"""

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


# ============================================================
# 지역 코드 매핑 (모듈 레벨 상수)
# ============================================================

# 소매 시장 지역 (24개)
RETAIL_REGIONS = {
    "서울": "1101",
    "부산": "2100",
    "대구": "2200",
    "인천": "2300",
    "광주": "2401",
    "대전": "2501",
    "울산": "2601",
    "수원": "3111",
    "강릉": "3214",
    "춘천": "3211",
    "청주": "3311",
    "전주": "3511",
    "포항": "3711",
    "제주": "3911",
    "의정부": "3113",
    "순천": "3613",
    "안동": "3714",
    "창원": "3814",
    "용인": "3145",
    "세종": "2701",
    "성남": "3112",
    "고양": "3138",
    "천안": "3411",
    "김해": "3818",
}

# 도매 시장 지역 (5개)
WHOLESALE_REGIONS = {
    "서울": "1101",
    "부산": "2100",
    "대구": "2200",
    "광주": "2401",
    "대전": "2501",
}

# 지역명 별칭
REGION_ALIASES = {
    "서울시": "서울",
    "서울특별시": "서울",
    "부산시": "부산",
    "부산광역시": "부산",
    "대구시": "대구",
    "대구광역시": "대구",
    "인천시": "인천",
    "인천광역시": "인천",
    "광주시": "광주",
    "광주광역시": "광주",
    "대전시": "대전",
    "대전광역시": "대전",
    "울산시": "울산",
    "울산광역시": "울산",
    "세종시": "세종",
    "세종특별자치시": "세종",
    "제주도": "제주",
    "제주특별자치도": "제주",
}


def _map_region(name: str, market_type: Optional[str] = None) -> Dict:
    """
    지역명을 지역코드로 매핑

    Args:
        name: 지역명 (예: "서울", "서울시", "부산")
        market_type: "소매" 또는 "도매"

    Returns:
        지역 정보 딕셔너리
    """
    # 시장 타입 결정
    mtype = market_type or "소매"

    # 지역명 정규화 (별칭 처리)
    normalized = REGION_ALIASES.get(name.strip(), name.strip())

    # 매핑 테이블 선택
    mapping = RETAIL_REGIONS if mtype == "소매" else WHOLESALE_REGIONS

    # 매핑 시도
    if normalized in mapping:
        return {
            "name": normalized,
            "code": mapping[normalized],
        }

    # Fallback: 도매에서 못 찾았으면 소매에 있는지 확인
    if mtype == "도매" and normalized in RETAIL_REGIONS:
        return {
            "code": None,
            "name": normalized,
            "retail_code": RETAIL_REGIONS[normalized],
        }

    # 찾을 수 없음
    return {
        "code": None,
        "name": normalized,
        "error": "지역을 찾을 수 없습니다",
    }


# ============================================================
# LLM 추출 결과 모델
# ============================================================


class HierarchicalKeywords(BaseModel):
    """LLM이 추출한 계층별 키워드"""

    categories: List[str] = []
    products: List[str] = []
    kinds: List[str] = []
    grades: List[str] = []
    regions: List[str] = []
    market_type: List[str] = []


# LLM 프롬프트 (지역 포함)
SYSTEM_PROMPT = """너는 농축수산물 유통 정보 시스템의 검색 어시스턴트다.
사용자의 자연어 질문을 분석하여 다음 계층별로 키워드를 추출해야 한다:

1. 부류(category): 큰 분류 (예: 식량작물, 채소류, 특용작물, 과일류, 축산물, 수산물)
2. 품목(product): 구체적인 상품명 (예: 쌀, 배추, 사과, 돼지, 고등어)
3. 품종(kind): 품목의 세부 종류 (예: 캠벨얼리, 월동, 후지, 삼겹살, 안심)
4. 등급(grade): 품질 등급 (예: 상품, 중품, 하품, 특大, 大, 中, 小, 1등급, 1++등급)
5. 지역(region): 지역명 (예: 서울, 부산, 대구, 인천, 광주, 대전, 울산, 수원 등)
6. 시장구분(market_type): 소매 또는 도매

규칙:
- 품목, 품종, 등급 계층에서는 유사어/동의어를 포함해야 한다.
- 각 계층 정보가 명시되지 않으면 추측하지 말고 반드시 빈 리스트로 반환한다."""

USER_PROMPT = """다음 사용자 질문을 분석하여 계층별 키워드를 추출해줘.

질문: "{query}"

예시:

입력: "쌀 가격을 알려줘."
출력: {{"categories": ["식량작물"], "products": ["쌀"], "kinds": [], "grades": [], "regions": [], "market_type": []}}

입력: "후지 사과 가격 추이를 알려줘."
출력: {{"categories": ["과일류"], "products": ["사과"], "kinds": ["후지"], "grades": [], "regions": [], "market_type": []}}

입력: "취청 오이 상품 가격을 알려줘."
출력: {{"categories": ["채소류"], "products": ["오이"], "kinds": ["취청"], "grades": ["상품"], "regions": ["서울", "부산"], "market_type": []}}

입력: "서울과 대전 돼지고기 삼겹살 가격을 비교해줘."
출력: {{"categories": ["축산물"], "products": ["돼지고기", "돼지"], "kinds": ["삼겹살"], "grades": [], "regions": ["서울", "대전"], "market_type": []}}

입력: "작년 소 등심 마트 가격은 얼마야?"
출력: {{"categories": ["축산물"], "products": ["소고기", "소"], "kinds": ["등심"], "grades": [], "regions": [], "market_type": ["소매"]}}"""


class HierarchicalSearcher:
    """계층적 구조를 고려한 농축수산물 검색 (지역 포함)"""

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
        자연어 쿼리로 농축수산물 계층 정보 + 지역 정보 검색

        부류(category), 품목(product), 품종(kind), 등급(grade), 지역(region), 시장구분(market)의
        전체 계층 구조를 반환합니다.

        Args:
            natural_query: 자연어 검색어 (예: "서울 후지 사과 상품", "서울 부산 배추 비교")
            top_k: 최대 반환 개수

        Returns:
            계층 정보 리스트 (category, product, kind, grade, regions, market 포함)
        """
        # 1. LLM으로 계층별 키워드 추출 (지역 포함)
        keywords = self._extract_keywords(natural_query)
        logger.info(f"추출된 계층별 키워드: {keywords}")

        # 2. 지역 매핑 (지역명 → 지역코드)
        mapped_regions = []
        market_type = keywords.market_type[0] if keywords.market_type else None

        if keywords.regions:
            mapped_regions = [
                _map_region(region, market_type) for region in keywords.regions
            ]
            logger.info(f"매핑된 지역: {mapped_regions}")

        # 3. 계층적 검색 수행 (품목/품종/등급)
        raw_results = self._hierarchical_search(keywords)

        # 4. 결과 구조화 (category, product, kind, grade, regions, market)
        structured = self._structure_results(raw_results, mapped_regions, market_type)

        return structured[:top_k]

    def _extract_keywords(self, query: str) -> HierarchicalKeywords:
        """LLM으로 계층별 키워드 추출 (부류/품목/품종/등급/지역)"""
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
                regions=result.regions,
                market_type=result.market_type,
            )
        except Exception as e:
            logger.warning(f"LLM 키워드 추출 실패: {e}. 단순 정규화 사용")
            normalized = self.text_processor.normalize(query)
            return HierarchicalKeywords(products=[normalized])

    def _hierarchical_search(self, keywords: HierarchicalKeywords) -> List[Dict]:
        """계층적 검색 수행 (부류→품목→품종/등급 필터링)"""
        product_codes = set()

        # 1단계: 품목 검색
        if keywords.products:
            query, params = self.query_builder.build_product_search(keywords.products)
            rows = self.db.execute(query, tuple(params))
            product_codes = {row["product_code"] for row in rows}
            logger.info(f"품목 검색 결과: {len(product_codes)}개")

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

            logger.info(f"부류 필터링 후: {len(product_codes)}개")

        if not product_codes:
            logger.warning("검색된 품목 없음")
            return []

        # 3단계: 각 품목의 전체 계층 정보 조회 (품종/등급 포함)
        results = []
        for product_code in product_codes:
            items = self._get_hierarchy_info(product_code, keywords)
            results.extend(items)

        logger.info(f"최종 계층 정보: {len(results)}개")
        return results

    def _get_hierarchy_info(
        self, product_code: str, keywords: HierarchicalKeywords
    ) -> List[Dict]:
        """품목의 전체 계층 정보 조회 (부류/품목/품종/등급)"""
        # 축산물 여부 확인
        check_query = (
            "SELECT category_code FROM api_items WHERE product_code = ? LIMIT 1"
        )
        rows = self.db.execute(check_query, (product_code,))

        if not rows:
            return []

        is_livestock = rows[0]["category_code"] == self.LIVESTOCK_CATEGORY_CODE

        # 전체 계층 정보 조회 (품종/등급 필터링 포함)
        query, params = self.query_builder.build_hierarchy_search(
            product_code, keywords.kinds, keywords.grades, is_livestock
        )

        rows = self.db.execute(query, tuple(params))
        return [dict(row) for row in rows]

    def _structure_results(
        self,
        raw_results: List[Dict],
        mapped_regions: List[Dict],
        market_type: Optional[str],
    ) -> List[Dict]:
        """검색 결과를 구조화된 계층 형태로 변환 (category/product/kind/grade/regions/market)"""
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

            # 지역
            if mapped_regions:
                result["regions"] = mapped_regions

            # 시장 정보
            if market_type:
                result["market"] = {
                    "code": "02" if market_type == "도매" else "01",
                    "name": market_type,
                }
            else:
                result["market"] = {
                    "code": "01",
                    "name": "소매",
                }

            structured.append(result)

        return structured
