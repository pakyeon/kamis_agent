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
                description="품목/품종/등급 검색어 (예: '사과', '후지 사과', '배추 상품', '돼지고기 삼겹살 1등급')"
            )
            top_k: int = Field(default=3, ge=1, le=10, description="결과 개수")

        def search_item(natural_query: str, top_k: int = 3) -> Dict[str, Any]:
            """
            농축수산물 계층 정보 검색 (부류/품목/품종/등급)

            Args:
                natural_query: 검색어 (예: "사과", "후지 사과 상품", "돼지고기 삼겹살 1등급")
                top_k: 반환 개수

            Returns:
                계층 정보 리스트 (category, product, kind, grade 포함)
            """
            if not natural_query.strip():
                return {"error": "검색어 필요"}

            try:
                results = self.searcher.search(natural_query.strip(), top_k=top_k)
            except Exception as e:
                return {"error": str(e)}

            if not results:
                return {"candidates": [], "note": "결과없음"}

            # 결과 정리
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
                # 축산물과 일반 품목의 등급 코드가 다름
                grade_code = grade.get("code") or grade.get("productrank_code")
                grade_name = grade.get("name")

                candidate = {
                    "category_code": cat_code,
                    "category_name": cat_name,
                    "product_code": prod_code,
                    "product_name": prod_name,
                }

                if kind_code:
                    candidate["kind_code"] = kind_code
                    candidate["kind_name"] = kind_name

                if grade_code:
                    candidate["grade_code"] = grade_code
                    candidate["grade_name"] = grade_name

                candidates.append(candidate)

            usage_note = """
사용법:
- product_code → p_itemcode 또는 p_productno
- kind_code → p_kindcode  
- grade_code → p_productrankcode
- 품종/등급 코드를 API 파라미터로 전달하여 정확한 검색 수행
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
            description=(
                "농축수산물 계층 정보(부류/품목/품종/등급) 검색. "
                "품목명 인식 시 필수 호출. "
                "예: '사과', '후지 사과', '배추 상품', '돼지고기 삼겹살 1등급'"
            ),
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
