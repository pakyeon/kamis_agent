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

        # 1. 품목 검색 Tool
        tools.append(self._create_search_tool())

        # 2. API Tools
        for name, spec in API_ENDPOINTS.items():
            tools.append(self._create_api_tool(name, spec))

        return tools

    def _create_search_tool(self) -> StructuredTool:
        """품목 검색 Tool"""

        class SearchInput(BaseModel):
            natural_query: str = Field(description="품목명 또는 자연어")
            top_k: int = Field(default=3, ge=1, le=10, description="결과 개수")

        def search_item(natural_query: str, top_k: int = 3) -> Dict[str, Any]:
            """
            품목명을 품목코드로 변환

            Args:
                natural_query: 품목명 (예: "사과", "돼지고기")
                top_k: 반환 개수

            Returns:
                품목 정보 리스트
            """
            if not natural_query.strip():
                return {"error": "검색어 필요"}

            try:
                results = self.searcher.search(natural_query.strip(), top_k=top_k)
            except Exception as e:
                return {"error": str(e)}

            if not results:
                return {"candidates": [], "note": "결과없음"}

            # 중복 제거 및 정리
            candidates = []
            seen = set()

            for item in results:
                prod = item.get("product", {})
                code = prod.get("code")
                name = prod.get("name")
                key = f"{code}_{name}"

                if key not in seen:
                    seen.add(key)
                    candidates.append(
                        {
                            "product_code": code,
                            "product_name": name,
                            "category": item.get("category"),
                            "kind": item.get("kind"),
                            "grade": item.get("grade"),
                        }
                    )

            return {
                "candidates": candidates,
                "note": f"{len(candidates)}개 발견. product_code를 p_itemcode로 사용",
            }

        return StructuredTool.from_function(
            name="search_item",
            func=search_item,
            args_schema=SearchInput,
            description="품목명→코드변환. 품목명 나오면 필수호출. 예:사과,배추,돼지고기,고등어",
        )

    def _create_api_tool(self, name: str, spec: Dict[str, Any]) -> StructuredTool:
        """API Tool 생성"""

        # Pydantic 모델 동적 생성
        fields = {}
        for param_name, param_spec in spec.get("parameters", {}).items():
            fields[param_name] = (
                Optional[str],
                Field(default=None, description=param_spec.get("description", "")),
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
        param_list = ", ".join(spec.get("parameters", {}).keys()) or "없음"

        return StructuredTool.from_function(
            name=name,
            func=api_call,
            args_schema=InputModel,
            description=f"{spec['description']}. 파라미터: {param_list}",
        )
