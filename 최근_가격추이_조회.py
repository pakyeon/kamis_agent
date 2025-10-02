from __future__ import annotations

import os
import requests
import json
from typing import Dict, Any, Sequence, Annotated, Literal, List, Optional, TypeAlias
from datetime import datetime, timedelta
from typing_extensions import TypedDict
from logging import getLogger, INFO, basicConfig

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.prebuilt import ToolNode
from langchain.tools import tool

# Pydantic
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# 품목 검색 모듈
from llm_product_searcher import LLMProductSearcher

load_dotenv()
basicConfig(level=INFO)
log = getLogger(__name__)

# ======================
# 타입 별칭 정의
# ======================
ProductCode: TypeAlias = str
ProductName: TypeAlias = str
ProductPair: TypeAlias = Dict[Literal["product_code", "product_name"], str]
SelectionMode: TypeAlias = Literal["strict", "related", "all"]
FilterResult: TypeAlias = Dict[str, Any]

# ======================
# 환경 변수
# ======================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
KAMIS_CERT_KEY = os.environ.get("KAMIS_API_KEY")
KAMIS_CERT_ID = os.environ.get("KAMIS_CERT_ID")
DB_PATH = os.getenv("DB_PATH", "kamis_api_list.db")

KAMIS_URL = "http://www.kamis.or.kr/service/price/xml.do?action=recentlyPriceTrendList"

# ======================
# 상수(설정) 통합
# ======================
SELECTION_LIMITS: Dict[str, int] = {"strict": 1, "related": 3, "all": 5}
# LLM 호출 필요성 판단 임계값(후보 수 기준). 'all'은 항상 LLM 스킵(원 의도 유지)
FILTER_THRESHOLDS: Dict[str, int] = {"strict": 3, "related": 5}
API_TIMEOUT = 10


def _apply_mode_limit(
    items: List[Any], mode: Literal["strict", "related", "all"]
) -> List[Any]:
    """모드별 개수 제한 적용"""
    return items[: SELECTION_LIMITS[mode]]


# ======================
# 전역 검색기 & 필터링 LLM
# ======================
product_searcher = LLMProductSearcher(DB_PATH)

# 내부 필터링용 LLM (경량 모델 사용)
filter_llm = ChatOpenAI(
    model="gpt-5-mini",
    temperature=0,
    api_key=OPENAI_API_KEY,
    reasoning_effort="minimal",
)

# 길이가 1 이상인 int 리스트 타입
Indices = Annotated[List[int], Field(min_length=1)]


# ======================
# 구조화 출력 스키마 (with_structured_output용)
# ======================
class FilterSelection(BaseModel):
    indices: Indices = Field(..., description="선택된 후보의 0-기반 인덱스 목록")
    reason: str = Field("", description="선택 이유를 간단히 요약")


# LLM → 구조화 출력 바인딩 (핵심)
structured_filter_llm = filter_llm.with_structured_output(FilterSelection)


# ======================
# Agent State
# ======================
class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]


# ======================
# 정적 스키마 정의
# ======================
class KamisPriceQuery(BaseModel):
    """KAMIS 농축수산물 가격 조회 파라미터 (정적 스키마)"""

    user_query: str = Field(
        description=(
            "사용자의 원본 질의 (예: '배추 가격', '배추랑 무 시세')\n"
            "자동으로 품목을 검색하여 조회합니다."
        )
    )

    selection_mode: SelectionMode = Field(
        default="strict",
        description=(
            "품목 선택 모드:\n"
            "- strict: 가장 관련성 높은 1개만 조회 (기본값)\n"
            "- related: 관련 품목 포함하여 최대 3개 조회\n"
            "- all: 검색된 모든 품목 조회 (최대 5개)\n"
            "\n"
            "판단 기준:\n"
            "- 단일 품목 질의 (예: '배추 가격') → strict\n"
            "- 복수 품목 질의 (예: '배추랑 무') → related\n"
            "- 포괄적 질의 (예: '채소 가격') → all"
        ),
    )

    p_regday: str = Field(
        default="",
        description=(
            f"조회 날짜 (YYYY-MM-DD):\n"
            f"- 오늘: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"- 어제: {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}\n"
            "- 미입력시: 가장 최근 데이터 자동 조회"
        ),
    )

    p_returntype: Literal["json", "xml"] = Field(
        default="json", description="응답 형식 (json 또는 xml)"
    )


# ======================
# 적응형 필터링 함수
# ======================
def _should_use_llm_filtering(
    candidates: List[ProductPair],
    mode: SelectionMode,
) -> bool:
    """
    LLM 필터링 필요 여부를 판단합니다.
    후보가 적으면 필터링 스킵. 'all'은 항상 스킵(원래 의도 유지).
    """
    if mode == "all":
        return False
    return len(candidates) > FILTER_THRESHOLDS.get(mode, 0)


def _simple_select_products(
    candidates: List[ProductPair],
    mode: SelectionMode,
) -> List[ProductPair]:
    """LLM 없이 간단한 규칙으로 품목 선택"""
    selected = _apply_mode_limit(candidates, mode)
    log.info(f"[간단 선택] {mode} 모드 - {len(selected)}개 선택 (필터링 스킵)")
    return selected


def _smart_filter_products(
    user_query: str,
    candidates: List[ProductPair],
    mode: SelectionMode,
) -> List[ProductPair]:
    """
    품목 후보를 지능적으로 필터링합니다.
    - 후보가 적으면 LLM 호출 스킵하여 속도 향상
    - 후보가 많으면 LLM이 정확 품목만 선택
    """
    if not candidates:
        return []

    # 적응형 판단: 필터링이 필요한지 확인
    if not _should_use_llm_filtering(candidates, mode):
        return _simple_select_products(candidates, mode)

    # LLM 필터링 수행
    top_candidates = candidates[:10]  # 토큰 절약
    candidates_text = "\n".join(
        f"{i}. {c['product_name']}" for i, c in enumerate(top_candidates)
    )

    # 구조화 출력 프롬프트(간결)
    filter_prompt = f"""아래 후보들 중에서 질의에 맞는 품목 인덱스만 반환하세요.
- 출력은 반드시 스키마를 만족하는 JSON 객체여야 합니다.
- 선택 규칙:
  1) 품목만 선택 (가공품/부위/브랜드 제외)
  2) 다른 품목 제외 (배추≠양배추)
  3) 선택 개수: strict=1, related=최대3 (현재 모드={mode})

질의: "{user_query}"

후보:
{candidates_text}
"""

    try:
        # 구조화 출력 사용(파싱 불필요)
        result: FilterSelection = structured_filter_llm.invoke(filter_prompt)
        indices = result.indices
        reason = result.reason or ""

        # 유효 인덱스만
        valid_indices = [i for i in indices if 0 <= i < len(top_candidates)]
        if not valid_indices:
            log.warning("[경고] 유효한 인덱스 없음. 첫 번째 후보 사용")
            return [top_candidates[0]]

        # 모드별 개수 제한
        limited_indices = _apply_mode_limit(
            valid_indices, "related" if mode == "related" else "strict"
        )
        selected = [top_candidates[i] for i in limited_indices]

        if reason:
            log.info(f"[LLM 필터링] {reason}")

        return selected

    except Exception as e:
        log.warning(f"[경고] 구조화 출력 실패: {e}. 첫 번째 후보 사용")
        return [top_candidates[0]]


# ======================
# API 파라미터/호출 헬퍼
# ======================
def _build_params(
    product: ProductPair,
    p_regday: str,
    p_returntype: Literal["json", "xml"],
) -> Dict[str, Any]:
    params = {
        # URL에 action이 이미 있으므로 여기선 중복 지정 지양
        "p_cert_key": KAMIS_CERT_KEY,
        "p_cert_id": KAMIS_CERT_ID,
        "p_returntype": p_returntype,
        "p_productno": product["product_code"],
    }
    if p_regday:
        params["p_regday"] = p_regday
    return params


def _call_kamis_api(product: ProductPair, params: Dict[str, Any]) -> Dict[str, Any]:
    """KAMIS API 단일 호출 래퍼"""
    try:
        resp = requests.get(KAMIS_URL, params=params, timeout=API_TIMEOUT)
        resp.raise_for_status()
        data = resp.json() if params.get("p_returntype") == "json" else resp.text
        return {
            "product_code": product["product_code"],
            "product_name": product["product_name"],
            "status": "success",
            "data": data,
        }
    except Exception as e:
        return {
            "product_code": product["product_code"],
            "product_name": product["product_name"],
            "status": "failed",
            "error": str(e),
        }


# ======================
# 응답 빌더
# ======================
def _build_response(
    user_query: str,
    selection_mode: SelectionMode,
    selected_products: List[ProductPair],
    results: List[Dict[str, Any]],
    product_pairs: List[ProductPair],
) -> Dict[str, Any]:
    # 기존 구현과 동일한 슬라이스 정책을 유지(기능 동일성 보장)
    remaining = product_pairs[len(selected_products) : len(selected_products) + 5]
    return {
        "query": user_query,
        "selection_mode": selection_mode,
        "selected_products": [p["product_name"] for p in selected_products],
        "results": results,
        "additional_matches": (
            [p["product_name"] for p in remaining] if remaining else None
        ),
        "total_found": len(product_pairs),
    }


# ======================
# Tool 구현 (정적 스키마)
# ======================
@tool("get_kamis_price_trends", args_schema=KamisPriceQuery)
def get_kamis_price_trends(
    user_query: str,
    selection_mode: SelectionMode = "strict",
    p_regday: str = "",
    p_returntype: Literal["json", "xml"] = "json",
) -> FilterResult:
    """
    KAMIS 농축수산물 가격 조회 도구 (정적 스키마 + 적응형 필터링)

    사용자 쿼리에서 자동으로 품목을 검색하고,
    적응형 로직으로 지능적으로 필터링하여 정확한 품목만 조회합니다.
    """
    # 1) 후보 검색 (Early return)
    try:
        product_pairs = product_searcher.get_name_code_pairs(user_query)
    except Exception as e:
        return {
            "error": "Search module error",
            "message": f"품목 검색 중 오류 발생: {e}",
        }
    if not product_pairs:
        return {
            "error": "No products found",
            "message": f"'{user_query}'에서 품목을 찾을 수 없습니다.",
            "suggestion": "다른 품목명으로 시도해보세요.",
        }

    # 2) 적응형 필터링 (지능형 선택)
    selected_products = _smart_filter_products(
        user_query, product_pairs, selection_mode
    )

    # 3) API 호출 (리스트 컴프리헨션)
    results = [
        _call_kamis_api(p, _build_params(p, p_regday, p_returntype))
        for p in selected_products
    ]

    # 4) 응답 조립
    return _build_response(
        user_query=user_query,
        selection_mode=selection_mode,
        selected_products=selected_products,
        results=results,
        product_pairs=product_pairs,
    )


# ======================
# 시스템 프롬프트
# ======================
SYSTEM_PROMPT = f"""KAMIS 농축수산물 가격 정보 조회 시스템

## 입력
농축수산물 가격 추이 관련 질문 (품목, 날짜 포함 가능)

## 처리
1. 질의 분석:
   - 품목 파악 (단일/복수 판단)
   - 날짜 추출 → YYYY-MM-DD (오늘: {datetime.now().strftime('%Y-%m-%d')})

2. selection_mode 결정:
   - 단일 품목 (예: "배추 가격") → "strict"
   - 복수 품목 (예: "배추랑 무") → "related"
   - 포괄적 질의 (예: "채소 가격") → "all"

3. get_kamis_price_trends 도구 호출
   - 자동으로 정확한 품목만 선택되어 조회됩니다

## 출력 형식
[조회 조건]
품목: {{선택된 품목들}}
날짜: {{조회날짜}}

[가격 정보]
{{데이터 요약}}

[참고사항]
필요 시 추가 설명

## 주의사항
- 조회 실패 시 오류 원인 명시
- 불필요한 인사말, 부연설명 생략
- 간결하고 정확한 정보 전달
"""


# ======================
# 그래프 구성
# ======================
def build_kamis_agent() -> StateGraph:
    """
    KAMIS 가격 조회 에이전트를 생성합니다.
    LangGraph를 사용하여 Agent-Tool 상호작용을 구성합니다.
    """
    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
        reasoning_effort="minimal",
    )

    tools = [get_kamis_price_trends]
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(AgentState)

    def agent_node(state: AgentState) -> Dict[str, List[AnyMessage]]:
        """Agent 노드: LLM이 사용자 질의를 처리하고 도구 호출 결정"""
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")

    def should_continue(state: AgentState) -> str:
        """도구 호출 여부를 판단하는 조건부 엣지"""
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and getattr(
            last_message, "tool_calls", None
        ):
            return "tools"
        return END

    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ======================
# 사용자 인터페이스
# ======================
def query_kamis(user_query: str, verbose: bool = False) -> str:
    """
    KAMIS 가격 조회를 실행합니다.
    """
    app = build_kamis_agent()

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_query)]

    result = app.invoke({"messages": messages})

    if verbose:
        print("\n" + "=" * 80)
        print("=== 전체 대화 히스토리 ===")
        print("=" * 80)
        for i, msg in enumerate(result["messages"], 1):
            print(f"\n[{i}] {msg.__class__.__name__}")
            print("-" * 40)
            if hasattr(msg, "content"):
                print(msg.content)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                print(f"\nTool Calls: {msg.tool_calls}")
        print("\n" + "=" * 80 + "\n")

    return result["messages"][-1].content


# ======================
# 테스트
# ======================
if __name__ == "__main__":
    print("🌾 KAMIS 농축수산물 가격 조회 에이전트")
    print("=" * 80 + "\n")

    # 테스트 케이스
    test_query = "최근 배추 가격 추이를 알려줘"
    print(f"질문: {test_query}")
    print("-" * 80)
    try:
        answer = query_kamis(test_query, verbose=False)
        print(f"답변:\n{answer}")
    except Exception as e:
        print(f"오류 발생: {e}")
    print("\n" + "=" * 80 + "\n")
