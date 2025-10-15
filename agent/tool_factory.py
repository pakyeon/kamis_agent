# -*- coding: utf-8 -*-
"""LangChain Tool 생성"""

import datetime as dt
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, create_model
from langchain_core.tools import StructuredTool

from ..core.client import KamisClient
from ..search import HierarchicalSearcher
from .api_endpoints import API_ENDPOINTS


def validate_dates(params: Dict[str, Any]) -> Optional[str]:
    """날짜 검증"""
    for key in ("p_regday", "p_startday", "p_endday"):
        if params.get(key):
            try:
                dt.datetime.strptime(params[key], "%Y-%m-%d")
            except ValueError:
                return f"{key}는 YYYY-MM-DD 형식이어야 합니다"

    if params.get("p_startday") and params.get("p_endday"):
        if params["p_startday"] > params["p_endday"]:
            return "시작일이 종료일보다 늦습니다"

    return None


class ToolFactory:
    """LangChain Tool 생성 팩토리"""

    def __init__(self, client: KamisClient, searcher: HierarchicalSearcher):
        self.client = client
        self.searcher = searcher

    def create_all_tools(self) -> List[StructuredTool]:
        """모든 Tool 생성"""
        tools = []

        # 1. 계층 정보 검색 Tool
        tools.append(self._create_search_tool())

        # 2. API Tools
        for name, spec in API_ENDPOINTS.items():
            tools.append(self._create_api_tool(name, spec))

        return tools

    def _create_search_tool(self) -> StructuredTool:
        """농축수산물 계층 정보 검색 Tool"""

        class SearchInput(BaseModel):
            natural_query: str = Field(
                description="품목/품종/등급/지역/시장 검색어 (예: '사과', '후지 사과', '서울 배추 상품', '서울 부산 돼지고기 도매')"
            )
            top_k: int = Field(default=3, ge=1, le=10, description="결과 개수")

        def search_item(natural_query: str, top_k: int = 3) -> Dict[str, Any]:
            """
            농축수산물 계층 정보 검색 (부류/품목/품종/등급/지역/시장)

            Args:
                natural_query: 검색어 (예: "사과", "서울 후지 사과 상품", "서울 부산 돼지고기 삼겹살 1등급 도매")
                top_k: 반환 개수

            Returns:
                계층 정보 리스트 (category, product, kind, grade, region, market 포함)
            """
            if not natural_query.strip():
                return {"error": "검색어 필요"}

            try:
                results = self.searcher.search(natural_query.strip(), top_k=top_k)
            except Exception as e:
                return {"error": str(e)}

            if not results:
                return {"candidates": [], "note": "결과없음"}

            # 결과 정리 (평탄화)
            candidates = []

            for item in results:
                # 부류
                cat = item.get("category", {})
                cat_code = cat.get("code")
                cat_name = cat.get("name")

                # 품목
                prod = item.get("product", {})
                prod_code = prod.get("code")
                prod_name = prod.get("name")

                # 품종
                kind = item.get("kind", {})
                kind_code = kind.get("code")
                kind_name = kind.get("name")

                # 등급
                grade = item.get("grade", {})

                # 지역 (리스트)
                regions = item.get("regions", [])

                # 시장
                market = item.get("market", {})

                # 기본 정보
                candidate = {
                    "category_code": cat_code,
                    "category_name": cat_name,
                    "product_code": prod_code,
                    "product_name": prod_name,
                }

                # 품종 정보
                if kind_code:
                    candidate["kind_code"] = kind_code
                    candidate["kind_name"] = kind_name

                # 등급 정보 (축산물 vs 일반품목)
                if grade:
                    candidate["grade_name"] = grade.get("name")

                    # 축산물: code 필드
                    if grade.get("code"):
                        candidate["grade_code"] = grade.get("code")

                    # 일반품목: productrank_code, graderank_code
                    if grade.get("productrank_code"):
                        candidate["grade_productrank_code"] = grade.get(
                            "productrank_code"
                        )
                    if grade.get("graderank_code"):
                        candidate["grade_graderank_code"] = grade.get("graderank_code")

                # 지역 정보 평탄화
                if regions:
                    candidate["region_codes"] = [
                        r.get("code") for r in regions if r.get("code")
                    ]
                    candidate["region_names"] = [r.get("name") for r in regions]

                    # 에러 정보가 있는 지역
                    errors = [r.get("error") for r in regions if r.get("error")]
                    if errors:
                        candidate["region_errors"] = errors

                    # 소매 전용 코드 (도매 시장에 없는 지역)
                    retail_codes = [
                        r.get("retail_code") for r in regions if r.get("retail_code")
                    ]
                    if retail_codes:
                        candidate["region_retail_codes"] = retail_codes

                # 시장 정보
                if market.get("code"):
                    candidate["market_code"] = market.get("code")
                    candidate["market_name"] = market.get("name")

                candidates.append(candidate)

            usage_note = """
# 반환 필드 → API 파라미터 매핑

계층 정보:
  product_code → p_itemcode 또는 p_productno
  kind_code → p_kindcode
  category_code → p_itemcategorycode 또는 p_item_category_code

등급 코드 (축산물 vs 일반품목):
  축산물(category_code=500):
    grade_code → p_productrankcode
  일반품목:
    grade_productrank_code → p_productrankcode
    grade_graderank_code → p_graderank (monthly_sales, yearly_sales만)

지역/시장:
  region_codes[i] → p_countrycode (각 지역별 개별 호출)
  market_code → p_productclscode 또는 p_product_cls_code

필드 설명:
  region_codes: 조회 가능한 지역코드 리스트
  region_names: 지역명 (참고용)
  region_errors: 매핑 실패한 지역 (있는 경우)
  market_name: "소매" 또는 "도매" (참고용)
            """.strip()

            return {
                "candidates": candidates,
                "count": len(candidates),
                "note": usage_note,
            }

        return StructuredTool.from_function(
            name="search_item",
            func=search_item,
            args_schema=SearchInput,
            description=("농축수산물 계층 정보(부류/품목/품종/등급/지역/시장) 검색"),
        )

    def _create_api_tool(self, name: str, spec: Dict[str, Any]) -> StructuredTool:
        """API Tool 생성"""

        # Pydantic 모델 동적 생성
        fields = {}
        for param_name, param_spec in spec.get("fields", {}).items():
            if isinstance(param_spec, dict):
                desc = param_spec.get("desc", "")
            else:
                desc = param_spec or ""

            fields[param_name] = (
                Optional[str],
                Field(default=None, description=desc),
            )

        InputModel = create_model(f"{name}_Input", **fields)

        def api_call(**kwargs) -> Dict[str, Any]:
            """API 호출"""
            # None 제거
            params = {k: v for k, v in kwargs.items() if v is not None}

            # 날짜 검증
            if error := validate_dates(params):
                return {"error": error}

            # API 호출
            try:
                response = self.client.call(spec["action"], params)
                return {
                    "action": spec["action"],
                    "params": params,
                    "response": response,
                }
            except Exception as e:
                return {"error": str(e)}

        # 파라미터 목록 생성
        param_list = ", ".join(spec.get("fields", {}).keys()) or "없음"

        return StructuredTool.from_function(
            name=name,
            func=api_call,
            args_schema=InputModel,
            description=f"{spec['desc']}. 파라미터: {param_list}",
        )
