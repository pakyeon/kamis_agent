from __future__ import annotations
import os, requests
from typing import Optional, Dict, Any, Sequence, Annotated, Literal
from datetime import datetime
from pydantic import BaseModel, Field

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from dotenv import load_dotenv

load_dotenv()

# 환경 변수
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
KAMIS_CERT_KEY = os.environ.get("KAMIS_API_KEY")
KAMIS_CERT_ID = os.environ.get("KAMIS_CERT_ID")

KAMIS_URL = "http://www.kamis.co.kr/service/price/xml.do?action=dailySalesList"

# ===========================================
# Tool Schema 정의
# ===========================================


class KamisRecentSalesQuery(BaseModel):
    """KAMIS 최근일자 도소매가격정보 조회 파라미터 (상품 기준)"""

    p_returntype: Literal["json", "xml"] = Field(
        "json",
        description=(
            "응답 형식:\n"
            "- 'json': JSON 데이터 형식\n"
            "- 'xml': XML 데이터 형식\n"
            "기본값: 'json'"
        ),
    )

    # API 인증 정보 (내부적으로 자동 설정)
    p_cert_key: str = Field(default="", exclude=True)
    p_cert_id: str = Field(default="", exclude=True)


# ===========================================
# Tool 구현
# ===========================================


@tool("get_kamis_recent_sales", args_schema=KamisRecentSalesQuery)
def get_kamis_recent_sales(
    p_returntype: Literal["json", "xml"] = "json",
) -> Dict[str, Any]:
    """
    KAMIS(농산물유통정보) API를 통해 최근일자 도소매 가격 정보를 조회합니다.

    이 API는 품목별(상품 기준) 최신 가격 정보를 제공합니다.
    - 카테고리별 조회가 아닌 개별 품목별 최신 가격
    - 도매/소매 가격이 함께 제공됨
    - 별도의 날짜, 지역 지정 없이 최신 데이터 제공

    사용 시점:
    - 사용자가 특정 품목의 "최신" 또는 "오늘" 가격을 요청할 때
    - 날짜나 지역 지정 없이 전반적인 가격 현황을 원할 때
    - 여러 품목의 현재 시세를 비교하고 싶을 때

    참고: 특정 날짜나 지역별 상세 조회가 필요한 경우
    get_kamis_price 도구를 사용하세요.
    """
    # API 인증 정보 자동 설정
    params = {
        "action": "dailySalesList",
        "p_cert_key": KAMIS_CERT_KEY,
        "p_cert_id": KAMIS_CERT_ID,
        "p_returntype": p_returntype,
    }

    try:
        response = requests.get(KAMIS_URL, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e), "message": "API 호출 실패"}


# ===========================================
# Agent 구성
# ===========================================


class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]


SYSTEM_PROMPT = f"""KAMIS 최근일자 도소매가격 조회 서브시스템.

## 입력
농산물 최신 가격 정보 조회 요청 (특정 품목명 또는 전체 현황)

## 기능 특성
- **최신 데이터 조회**: 가장 최근 업데이트된 가격 정보 제공
- **품목별 조회**: 개별 상품 기준 (카테고리 전체가 아님)
- **도소매 통합**: 도매가/소매가 함께 제공
- **날짜/지역 미지정**: 시스템에서 최신 가능 데이터 자동 선택

## 처리
get_kamis_recent_sales 도구로 최신 가격 정보 조회

## 출력 형식
```
[조회 정보]
조회 시점: {{API 반환 날짜}}
데이터 특성: 최근일자 기준 전국 평균

[가격 현황]
{{데이터}}

[참고사항]
{{추가 설명}}
```

## 사용자 질의 유형별 처리
1. "오늘 시금치 가격" → 최신 데이터에서 시금치 검색
2. "현재 채소 가격 현황" → 전체 데이터에서 채소류 필터링
3. "최근 가격 알려줘" → 전체 데이터 요약 제공

조회 실패시 오류 원인 명시. 불필요한 인사말, 부연설명 생략."""


def build_kamis_recent_sales_agent():
    """KAMIS 최근 가격 조회 에이전트 생성"""

    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
        reasoning_effort="minimal",
    )
    tools = [get_kamis_recent_sales]
    llm_with_tools = llm.bind_tools(tools)

    graph = StateGraph(AgentState)

    def agent_node(state: AgentState):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        return END

    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ===========================================
# 사용자 인터페이스
# ===========================================


def query_kamis_recent_sales(user_query: str, verbose: bool = False) -> str:
    """
    KAMIS 최근 가격 조회 실행

    Args:
        user_query: 사용자 질의 (예: "오늘 시금치 가격 알려줘")
        verbose: True시 전체 메시지 히스토리 출력

    Returns:
        구조화된 최신 가격 정보 텍스트
    """
    app = build_kamis_recent_sales_agent()

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_query)]

    result = app.invoke({"messages": messages})

    if verbose:
        print("=== 전체 대화 히스토리 ===")
        for msg in result["messages"]:
            print(f"\n[{msg.__class__.__name__}]")
            print(msg.content if hasattr(msg, "content") else msg)
        print("\n" + "=" * 50 + "\n")

    return result["messages"][-1].content


# ===========================================
# 테스트
# ===========================================

if __name__ == "__main__":
    # 테스트 케이스
    test_queries = "최근 시금치 가격 알려줘"

    print("🌾 KAMIS 최근일자 가격 조회 에이전트 테스트\n")

    query = test_queries
    print(f"질문: {query}")
    print(f"답변:\n{query_kamis_recent_sales(query)}")
    print("-" * 80)
