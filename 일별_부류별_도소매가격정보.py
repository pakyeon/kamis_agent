from __future__ import annotations
import os, requests
from typing import Optional, Dict, Any, Sequence, Annotated, Literal
from datetime import datetime, timedelta
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

KAMIS_URL = (
    "http://www.kamis.or.kr/service/price/xml.do?action=dailyPriceByCategoryList"
)

# ===========================================
# Tool Schema 정의
# ===========================================


class KamisPriceQuery(BaseModel):
    """KAMIS 농산물 가격 조회 파라미터"""

    p_product_cls_code: Literal["01", "02"] = Field(
        description=(
            "판매 대상 구분:\n"
            "- '01': 소매가격 (마트, 소매점의 소비자 판매가)\n"
            "- '02': 도매가격 (가락시장, 공판장 거래가)\n"
            "기본값: '02'"
        ),
        default="02",
    )

    p_item_category_code: Literal["100", "200", "300", "400", "500", "600"] = Field(
        description=(
            "품목 카테고리:\n"
            "- '100': 식량작물 (쌀, 밀, 콩 등)\n"
            "- '200': 채소류 (배추, 무, 양파 등)\n"
            "- '300': 특용작물 (버섯, 인삼 등)\n"
            "- '400': 과일류 (사과, 배, 포도 등)\n"
            "- '500': 축산물 (소고기, 돼지고기, 계란 등)\n"
            "- '600': 수산물 (생선, 새우, 오징어 등)\n"
            "기본값: '100'"
        ),
        default="100",
    )

    p_country_code: Optional[str] = Field(
        None,
        description=(
            "지역 코드 (선택사항, 미입력시 전국 평균):\n\n"
            "소매가격(01) 가능 지역:\n"
            "- '1101': 서울, '2100': 부산, '2200': 대구, '2300': 인천\n"
            "- '2401': 광주, '2501': 대전, '2601': 울산, '2701': 세종\n"
            "- '3111': 수원, '3112': 성남, '3113': 의정부, '3138': 고양, '3145': 용인\n"
            "- '3211': 춘천, '3214': 강릉, '3311': 청주, '3411': 천안\n"
            "- '3511': 전주, '3613': 순천, '3711': 포항, '3714': 안동\n"
            "- '3818': 김해, '3814': 창원, '3911': 제주\n\n"
            "도매가격(02) 가능 지역:\n"
            "- '1101': 서울, '2100': 부산, '2200': 대구\n"
            "- '2401': 광주, '2501': 대전\n\n"
            "주의: 소매/도매 구분에 따라 선택 가능한 지역이 다름!"
        ),
    )

    p_regday: str = Field(
        description=(
            "조회 날짜 (YYYY-MM-DD 형식, 선택사항):\n"
            f"- 오늘: {datetime.now().strftime('%Y-%m-%d')}\n"
            f"- 어제: {(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')}\n"
            "- 미입력시: 가장 최근 데이터 자동 조회\n"
        ),
        default_factory=datetime.now().strftime("%Y-%m-%d"),
    )

    p_convert_kg_yn: Literal["Y", "N"] = Field(
        "N",
        description=(
            "kg 단위 환산 여부:\n"
            "- 'Y': kg 기준으로 환산하여 표시\n"
            "- 'N': 원래 거래 단위 그대로 표시\n"
            "기본값: 'N'"
        ),
    )

    p_returntype: Literal["json", "xml"] = Field(
        "json", description="응답 형식 (json 또는 xml, 기본값: json)"
    )

    # API 인증 정보 (내부적으로 자동 설정)
    p_cert_key: str = Field(default="", exclude=True)
    p_cert_id: str = Field(default="", exclude=True)


# ===========================================
# Tool 구현
# ===========================================


@tool("get_kamis_price", args_schema=KamisPriceQuery)
def get_kamis_price(
    p_product_cls_code: Literal["01", "02"] = "02",
    p_item_category_code: Literal["100", "200", "300", "400", "500", "600"] = "100",
    p_country_code: Optional[str] = None,
    p_regday: str = datetime.now().strftime("%Y-%m-%d"),
    p_convert_kg_yn: Literal["Y", "N"] = "N",
    p_returntype: Literal["json", "xml"] = "json",
) -> Dict[str, Any]:
    """
    KAMIS(농산물유통정보) API를 통해 농산물 일별 가격 정보를 조회합니다.

    사용자 질의를 분석하여 적절한 파라미터를 선택하세요:
    - 품목 종류에 맞는 카테고리 코드
    - 소비자가/시장가 구분 (소매/도매)
    - 지역 (소매/도매별 가능 지역 확인 필수)
    - 날짜 (미지정시 오늘로 지정)

    반환 데이터가 없거나 '-'인 경우, 날짜를 조정하여 재시도하세요.
    """
    # API 인증 정보 자동 설정
    params = {
        "action": "dailyPriceByCategoryList",
        "p_cert_key": KAMIS_CERT_KEY,
        "p_cert_id": KAMIS_CERT_ID,
        "p_product_cls_code": p_product_cls_code,
        "p_item_category_code": p_item_category_code,
        "p_returntype": p_returntype,
        "p_convert_kg_yn": p_convert_kg_yn,
    }

    if p_country_code:
        params["p_country_code"] = p_country_code
    if p_regday:
        params["p_regday"] = p_regday

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


SYSTEM_PROMPT = f"""KAMIS 농산물 가격 조회 서브시스템.

## 입력
농산물 가격 관련 질의 (품목, 지역, 날짜 등 포함 가능)

## 처리
1. 질의 분석:
   - 품목 → 카테고리 코드 (식량작물/채소/과일/축산/수산 등)
   - 가격 유형 → 소매(01: 마트/소매점) 또는 도매(02: 시장/공판장)
   - 지역 → 지역 코드 (서울:1101, 부산:2100 등)
   - 날짜 → YYYY-MM-DD (오늘: {datetime.now().strftime('%Y-%m-%d')})

2. get_kamis_price 도구로 조회
   - 소매/도매별 가능 지역 확인 필수
   - 데이터 없으면 날짜 조정 후 재시도

## 출력 형식
```
[조회 조건]
품목: {{카테고리명}}
유형: {{소매/도매}}
지역: {{지역명}} (또는 전국)
날짜: {{조회날짜}}

[가격 정보]
{{데이터}}

[참고사항]
{{필요시 추가 설명}}
```

조회 실패시 오류 원인 명시. 불필요한 인사말, 부연설명 생략."""


def build_kamis_agent():
    """KAMIS 조회 에이전트 생성"""

    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=0,
        api_key=OPENAI_API_KEY,
        reasoning_effort="minimal",
    )
    tools = [get_kamis_price]
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


def query_kamis(user_query: str, verbose: bool = False) -> str:
    """
    KAMIS 가격 조회 실행

    Args:
        user_query: 사용자 질의 (예: "서울 마트 배추 가격 알려줘")
        verbose: True시 전체 메시지 히스토리 출력

    Returns:
        구조화된 가격 정보 텍스트
    """
    app = build_kamis_agent()

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
    test_queries = "서울 마트에서 파는 배추 가격 알려줘"

    print("🌾 KAMIS 농산물 가격 조회 에이전트 테스트\n")

    query = test_queries
    print(f"질문: {query}")
    print(f"답변:\n{query_kamis(query)}")
    print("-" * 80)
