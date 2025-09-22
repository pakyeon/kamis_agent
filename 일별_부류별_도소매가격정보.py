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
KAMIS_URL = (
    "http://www.kamis.or.kr/service/price/xml.do?action=dailyPriceByCategoryList"
)

# ===========================================
# 핵심 기능 1: 기본 정보 제공 (LLM 판단용 참고 자료)
# ===========================================

# 소매가격 지역 코드 정보
RETAIL_REGION_INFO = {
    "1101": "서울",
    "2100": "부산",
    "2200": "대구",
    "2300": "인천",
    "2401": "광주",
    "2501": "대전",
    "2601": "울산",
    "3111": "수원",
    "3214": "강릉",
    "3211": "춘천",
    "3311": "청주",
    "3511": "전주",
    "3711": "포항",
    "3911": "제주",
    "3113": "의정부",
    "3613": "순천",
    "3714": "안동",
    "3814": "창원",
    "3145": "용인",
    "2701": "세종",
    "3112": "성남",
    "3138": "고양",
    "3411": "천안",
    "3818": "김해",
}

# 도매가격 지역 코드 정보
WHOLESALE_REGION_INFO = {
    "1101": "서울",
    "2100": "부산",
    "2200": "대구",
    "2401": "광주",
    "2501": "대전",
}

# 카테고리 정보
CATEGORY_INFO = {
    "100": "식량작물 (쌀, 밀, 보리, 콩, 옥수수 등 곡물류)",
    "200": "채소류 (배추, 무, 양파, 당근, 시금치, 상추 등)",
    "300": "특용작물 (버섯, 인삼, 약용작물 등)",
    "400": "과일류 (사과, 배, 포도, 딸기, 바나나, 오렌지 등)",
    "500": "축산물 (소고기, 돼지고기, 닭고기, 계란, 우유 등)",
    "600": "수산물 (생선, 새우, 오징어, 명태, 고등어 등)",
}

# 소매/도매 정보
PRODUCT_CLS_INFO = {
    "01": "소매 (마트, 소매점에서 소비자에게 판매되는 가격)",
    "02": "도매 (가락시장, 공판장 등에서 거래되는 가격)",
}


@tool("kamis_param_infer")
def kamis_param_infer_tool(query: str) -> Dict[str, Any]:
    """
    사용자 자연어 쿼리 분석을 위한 기본 정보 제공
    LLM이 모든 파라미터를 직접 판단하도록 안내
    """
    today = datetime.now().date().strftime("%Y-%m-%d")

    guide = f"""
사용자 요청을 분석하여 KAMIS API 파라미터를 설정하세요.

=== 오늘 날짜 ===
오늘은 {today} 입니다.

=== 설정할 파라미터 ===

1. 카테고리 코드 (p_item_category_code):
{chr(10).join([f'   - {code}: {desc}' for code, desc in CATEGORY_INFO.items()])}

2. 소매/도매 구분 (p_product_cls_code):
{chr(10).join([f'   - {code}: {desc}' for code, desc in PRODUCT_CLS_INFO.items()])}

3. 지역 코드 (p_country_code) - 선택사항:
   - None 또는 생략: 전체지역 (기본값)
   
   소매가격 선택가능 지역:
{chr(10).join([f'     - {code}: {name}' for code, name in RETAIL_REGION_INFO.items()])}
   
   도매가격 선택가능 지역:
{chr(10).join([f'     - {code}: {name}' for code, name in WHOLESALE_REGION_INFO.items()])}
   
   주의: 
   - 전체지역 조회시 p_country_code를 None으로 설정하거나 생략하세요
   - 소매가격과 도매가격은 선택 가능한 지역이 다릅니다!

4. 조회 날짜 (p_regday) - 선택사항:
   - YYYY-MM-DD 형식으로 입력 (예: {today})
   - 오늘: {today}
   - 어제: {(datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")}
   - 미입력시 최신 가능한 날짜로 자동 조회

5. kg 환산 여부 (p_convert_kg_yn):
   - Y: kg 기준으로 환산하여 표시
   - N: 원래 단위 그대로 표시

사용자 요청: "{query}"

위 정보를 바탕으로 kamis_daily_price_by_category 도구를 적절한 파라미터와 함께 호출하세요.
소매/도매 구분에 따라 해당하는 지역만 선택할 수 있음에 주의하세요.
"""

    return {
        "guide": guide,
        "today": today,
        "available_categories": CATEGORY_INFO,
        "retail_regions": RETAIL_REGION_INFO,
        "wholesale_regions": WHOLESALE_REGION_INFO,
        "product_cls_options": PRODUCT_CLS_INFO,
        "_note": "LLM이 사용자 요청을 분석하여 적절한 파라미터로 kamis_daily_price_by_category를 호출해야 합니다. 소매/도매에 따라 지역 선택이 제한됩니다.",
    }


# ===========================================
# 핵심 기능 2: KAMIS API 호출
# ===========================================
class KamisParams(BaseModel):
    p_cert_key: str
    p_cert_id: str
    p_returntype: Literal["json"] = "json"
    p_product_cls_code: Literal["01", "02"] = "02"
    p_item_category_code: Literal["100", "200", "300", "400", "500", "600"] = "100"
    p_country_code: Optional[str] = None
    p_regday: Optional[str] = None
    p_convert_kg_yn: Literal["Y", "N"] = "N"


def call_kamis_api(params: KamisParams) -> Dict[str, Any]:
    """KAMIS API 호출"""
    query_params = params.model_dump(exclude_none=True)
    query_params["action"] = "dailyPriceByCategoryList"

    response = requests.get(KAMIS_URL, params=query_params)
    return response.json()


@tool("kamis_daily_price_by_category", args_schema=KamisParams)
def kamis_tool(**kwargs) -> Dict[str, Any]:
    """KAMIS 일별 가격 정보 조회"""
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
   - 카테고리: 어떤 농산물 분류에 해당하는가?
   - 소매/도매: 소비자 가격(소매) vs 시장 가격(도매)?
   - 지역: 특정 지역이 언급되었는가? (소매/도매별 지역 제한 확인)
   - 날짜: 언제 기준 가격인가? (오늘 날짜 정보 참고)
   - 단위: kg 기준 환산이 필요한가?

3. 판단한 결과로 kamis_daily_price_by_category 도구를 호출하세요.
4. 결과를 사용자에게 친화적으로 설명하세요.

중요 규칙: 
- 모든 파라미터는 사용자 요청과 컨텍스트를 바탕으로 직접 판단하세요.
- 소매가격과 도매가격은 조회 가능한 지역이 다릅니다! 반드시 확인하세요.
- 애매한 경우 기본 옵션을 선택하세요.
- 날짜가 명시되지 않으면 최신 데이터를 조회하도록 하세요.
- 데이터가 "-"인 경우 반드시 다른 시점의 데이터를 확인하여 가장 가까운 유효한 가격을 찾아서 답변하고, 다른 이유에 대해서 설명하세요.
- 지역 코드가 해당 소매/도매에서 지원되지 않는 경우, 사용자에게 알리고 대안을 제시하세요."""


def query_kamis(user_query: str):
    """KAMIS 조회 실행"""
    app = build_kamis_agent()

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_query)]

    result = app.invoke({"messages": messages})
    return result["messages"][-1].content


# 테스트 예시
if __name__ == "__main__":
    result = query_kamis("서울 마트에서 파는 배추 가격 알려줘")
    print(result)
