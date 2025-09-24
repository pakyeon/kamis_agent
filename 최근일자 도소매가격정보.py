from __future__ import annotations
import os, re, requests
from typing import Optional, Dict, Any, List, Sequence, Annotated, TypedDict, Literal
from datetime import datetime, timedelta
from dateutil.parser import parse as dtparse
from pydantic import BaseModel

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages, AnyMessage
from langgraph.prebuilt import ToolNode

# 환경 변수
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
KAMIS_CERT_KEY = os.environ.get("KAMIS_API_KEY")
KAMIS_CERT_ID = os.environ.get("KAMIS_CERT_ID")

# KAMIS API URL
KAMIS_URL = "http://www.kamis.co.kr/service/price/xml.do?action=dailySalesList"

# ===========================================
# 핵심 기능 1: 기본 정보 제공 (LLM 판단용 참고 자료)
# ===========================================


@tool("kamis_param_infer")
def kamis_param_infer_tool(query: str) -> Dict[str, Any]:

    today = datetime.now().date().strftime("%Y-%m-%d")

    """
    사용자 자연어 쿼리 분석을 위한 기본 정보 제공
    LLM이 모든 파라미터를 직접 판단하도록 안내
    """

    guide = f"""
사용자 요청을 분석하여 KAMIS API 파라미터를 설정하세요.

=== 오늘 날짜 ===
오늘은 {today} 입니다.

=== 설정할 파라미터 ===

1. 출력 형식 여부 (p_returntype):
   - json: Json 데이터 형식
   - xml: XML 데이터 형식

사용자 요청: "{query}"

위 정보를 바탕으로 kamis_daily_price_by_category 도구를 적절한 파라미터와 함께 호출하세요.
소매/도매 구분에 따라 해당하는 지역만 선택할 수 있음에 주의하세요.
"""

    return {
        "guide": guide,
        "_note": "LLM이 사용자 요청을 분석하여 적절한 파라미터로 kamis_daily_price_by_category를 호출해야 합니다.",
    }


# ===========================================
# 핵심 기능 2: KAMIS API 호출
# ===========================================
class KamisParams(BaseModel):
    p_cert_key: str
    p_cert_id: str
    p_returntype: Literal["xml", "json"] = "json"


def call_kamis_api(params: KamisParams) -> Dict[str, Any]:
    """KAMIS API 호출"""
    query_params = params.model_dump(exclude_none=True)
    query_params["action"] = "dailySalesList"

    response = requests.get(KAMIS_URL, params=query_params)
    return response.json()


@tool("kamis_daily_price_by_category", args_schema=KamisParams)
def kamis_tool(**kwargs) -> Dict[str, Any]:
    """KAMIS 최근일자 도소매가격정보(상품 기준)"""
    if not kwargs.get("p_cert_key"):
        kwargs["p_cert_key"] = KAMIS_CERT_KEY
    if not kwargs.get("p_cert_id"):
        kwargs["p_cert_id"] = KAMIS_CERT_ID

    params = KamisParams(**kwargs)
    result = call_kamis_api(params)
    return result


# ===========================================
# 핵심 기능 3: LLM Agent 구성
# ===========================================


class AgentState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]


def build_kamis_agent():
    """KAMIS 조회 에이전트 생성"""

    # LLM 설정
    llm = ChatOpenAI(model="gpt-5-mini", temperature=0)
    tools = [kamis_param_infer_tool, kamis_tool]
    llm_with_tools = llm.bind_tools(tools)

    # 그래프 생성
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
# 사용 예시
# ===========================================
SYSTEM_PROMPT = """당신은 KAMIS 농산물 가격 조회 도우미입니다.

사용자가 가격 조회를 요청하면 다음 단계를 따르세요:

1. 먼저 kamis_param_infer 도구를 호출하여 기본 정보와 가이드를 받으세요.
2. 사용자 질의를 분석하여 다음을 판단하세요:
    - 출력 형식: 어떤 형식으로 반환해야 하는가? 

3. 판단한 결과로 kamis_daily_price_by_category 도구를 호출하세요.
4. 결과를 사용자에게 친화적으로 설명하세요.

중요 규칙: 
- 모든 파라미터는 사용자 요청과 컨텍스트를 바탕으로 직접 판단하세요.
- 애매한 경우 기본 옵션을 선택하세요.
- 날짜가 명시되지 않으면 최신 데이터를 조회하도록 하세요.
- 사용자가 요청한 시점의 데이터가 "-"인 경우 반드시 최신 가격을 찾아서 답변하고, 시점이 다른 이유에 대해서 설명하세요."""


def query_kamis(user_query: str):
    """KAMIS 조회 실행"""
    app = build_kamis_agent()

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_query)]

    result = app.invoke({"messages": messages})
    return result["messages"][-1].content


# 테스트 예시
if __name__ == "__main__":
    result = query_kamis("오늘 시금치 가격을 알려줘.")
    print(result)
