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

KAMIS_URL = "http://www.kamis.or.kr/service/price/xml.do?action=dailyCountyList"

# ===========================================
# Tool Schema 정의
# ===========================================


class KamisCountyPriceQuery(BaseModel):
    """KAMIS 최근일자 지역별 도소매가격정보 조회 파라미터 (상품 기준)"""

    p_countycode: str = Field(
        "1101",
        description=(
            "지역 코드 (필수):\n\n"
            "소매가격 가능 지역:\n"
            "- '1101': 서울, '2100': 부산, '2200': 대구, '2300': 인천\n"
            "- '2401': 광주, '2501': 대전, '2601': 울산, '2701': 세종\n"
            "- '3111': 수원, '3112': 성남, '3113': 의정부, '3138': 고양, '3145': 용인\n"
            "- '3211': 춘천, '3214': 강릉, '3311': 청주, '3411': 천안\n"
            "- '3511': 전주, '3613': 순천, '3711': 포항, '3714': 안동\n"
            "- '3814': 창원, '3818': 김해, '3911': 제주\n\n"
            "도매가격 가능 지역:\n"
            "- '1101': 서울, '2100': 부산, '2200': 대구\n"
            "- '2401': 광주, '2501': 대전\n\n"
            "기본값: '1101' (서울)\n"
            "주의: 도매 조회시 도매 가능 지역만 선택 가능!"
        ),
    )

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


@tool("get_kamis_county_price", args_schema=KamisCountyPriceQuery)
def get_kamis_county_price(
    p_countycode: str = "1101",
    p_returntype: Literal["json", "xml"] = "json",
) -> Dict[str, Any]:
    """
    KAMIS API를 통해 특정 지역의 최근일자 도소매 가격 정보를 조회합니다.

    이 API는 지역별(시/도 단위) 품목별 최신 가격 정보를 제공합니다.
    - 특정 지역 기준 가격 조회
    - 도매/소매 가격이 함께 제공됨
    - 해당 지역의 최신 데이터만 제공

    사용 시점:
    - 사용자가 특정 지역의 가격을 요청할 때 ("서울", "부산" 등)
    - 지역별 가격 비교가 필요할 때
    - 특정 도시의 현재 시세를 알고 싶을 때

    참고:
    - 전국 평균이나 카테고리별 조회는 get_kamis_price 사용
    - 전체 지역 현황은 get_kamis_recent_sales 사용
    - 지역 코드는 도매/소매에 따라 선택 가능한 범위가 다름
    """
    # API 인증 정보 자동 설정
    params = {
        "action": "dailyCountyList",
        "p_cert_key": KAMIS_CERT_KEY,
        "p_cert_id": KAMIS_CERT_ID,
        "p_countycode": p_countycode,
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


SYSTEM_PROMPT = f"""KAMIS 지역별 도소매가격 조회 서브시스템.

## 입력
특정 지역의 농산물 가격 조회 요청 (지역명 포함)

## 기능 특성
- **지역 기준 조회**: 특정 시/도의 가격 정보
- **최신 데이터**: 해당 지역의 가장 최근 업데이트된 가격
- **도소매 통합**: 도매가/소매가 함께 제공
- **품목별 제공**: 해당 지역에서 거래되는 품목들

## 처리
1. 질의 분석:
   - 지역명 추출 → 지역 코드로 변환
     * 서울: 1101, 부산: 2100, 대구: 2200, 인천: 2300
     * 광주: 2401, 대전: 2501, 울산: 2601, 세종: 2701
     * 수원: 3111, 춘천: 3211, 청주: 3311, 전주: 3511
     * 제주: 3911, 기타 도시 코드는 tool description 참조
   
   - 지역 미지정시: 서울(1101) 기본값

2. get_kamis_county_price 도구로 조회
   - 도매가격 요청시 도매 가능 지역만 선택
   - 불가능한 지역인 경우 사용자에게 안내

## 출력 형식
```
[조회 정보]
지역: {{지역명}} ({{지역코드}})
조회 시점: {{API 반환 날짜}}

[가격 현황]
{{데이터}}

[참고사항]
{{추가 설명}}
```

## 지역별 처리 규칙
1. 소매가격: 모든 주요 도시 조회 가능
2. 도매가격: 서울, 부산, 대구, 광주, 대전만 가능
3. 사용자가 도매+지방 도시 요청시:
   - 불가능함을 안내
   - 가능한 대안 제시 (가까운 도매 가능 지역 또는 소매 가격)

조회 실패시 오류 원인 명시. 불필요한 인사말, 부연설명 생략."""


def build_kamis_county_agent():
    """KAMIS 지역별 가격 조회 에이전트 생성"""

    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
        reasoning_effort="minimal",
    )
    tools = [get_kamis_county_price]
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


def query_kamis_county_price(user_query: str, verbose: bool = False) -> str:
    """
    KAMIS 지역별 가격 조회 실행

    Args:
        user_query: 사용자 질의 (예: "부산 소매 시금치 가격 알려줘")
        verbose: True시 전체 메시지 히스토리 출력

    Returns:
        구조화된 지역별 가격 정보 텍스트
    """
    app = build_kamis_county_agent()

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
    test_queries = "가장 최근 부산 소매 시금치 가격 알려줘"

    print("🌾 KAMIS 지역별 가격 조회 에이전트 테스트\n")

    query = test_queries
    print(f"질문: {query}")
    print(f"답변:\n{query_kamis_county_price(query)}")
    print("-" * 80)
