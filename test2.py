#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KAMIS 멀티-툴 LLM 에이전트 (최적화 버전)
- KAMIS(한국농수산식품유통공사) Open API 17개 통합
- LangGraph + LangChain Tools
- 자연어 → 품목코드 자동 매핑
"""

import os
import sys
import json
import argparse
import datetime as dt
from typing import Any, Dict, List, Optional, TypedDict

import requests
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

from llm_product_searcher import LLMHierarchicalSearcher

load_dotenv()

# 환경변수
KAMIS_CERT_KEY = os.getenv("KAMIS_CERT_KEY")
KAMIS_CERT_ID = os.getenv("KAMIS_CERT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "kamis_api_list.db")


# =========================
# KAMIS 클라이언트
# =========================
class KamisClient:
    BASE_URL = "http://www.kamis.or.kr/service/price/xml.do"

    def __init__(self, cert_key: str, cert_id: str):
        self.cert_key = cert_key
        self.cert_id = cert_id
        self.sess = requests.Session()

    def call(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        q = {
            **params,
            "action": action,
            "p_cert_key": self.cert_key,
            "p_cert_id": self.cert_id,
            "p_returntype": "json",
        }
        try:
            resp = self.sess.get(self.BASE_URL, params=q, timeout=20)
            resp.raise_for_status()
            return resp.json() if resp.text else {"error": "응답 없음"}
        except requests.Timeout:
            return {"error": "타임아웃"}
        except requests.HTTPError as e:
            return {"error": f"HTTP {e.response.status_code}"}
        except Exception as e:
            return {"error": str(e)}


# =========================
# API 정의 (17개)
# =========================
API_DEFS = {
    "daily_by_category": {
        "action": "dailyPriceByCategoryList",
        "desc": "일별 부류별 가격. 부류코드(100=식량,200=채소,300=과일,400=특용,500=축산,600=수산) 전체 조회시",
        "fields": {
            "p_product_cls_code": "구분(01=소매,02=도매)",
            "p_item_category_code": "부류코드(100-600)",
            "p_country_code": "시군구코드",
            "p_regday": "조회일자 YYYY-MM-DD",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "daily_by_item_period": {
        "action": "periodProductList",
        "desc": "일별 품목별 가격(기간,최대1년). 특정품목 일자별추이. 필수:품목코드(search_item먼저)",
        "fields": {
            "p_startday": "시작일 YYYY-MM-DD",
            "p_endday": "종료일 YYYY-MM-DD",
            "p_productclscode": "구분(01=소매,02=도매)",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드(필수)",
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "monthly_sales": {
        "action": "monthlySalesList",
        "desc": "월별 평균가격. 월단위 집계",
        "fields": {
            "p_yyyy": "기준연도",
            "p_period": "조회기간(년수,기본3)",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드",
            "p_kindcode": "품종코드",
            "p_graderank": "등급(1=상,2=중,3=하)",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "yearly_sales": {
        "action": "yearlySalesList",
        "desc": "연별 평균가격. 1년초과 장기추이시 필수",
        "fields": {
            "p_yyyy": "기준연도",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드",
            "p_kindcode": "품종코드",
            "p_graderank": "등급(1=상,2=중,3=하)",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "old_eco_period": {
        "action": "periodNaturePriceList",
        "desc": "친환경가격(2005-2020.3). 기간별",
        "fields": {
            "p_startday": "시작일",
            "p_endday": "종료일",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드",
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "daily_sales_list": {
        "action": "dailySalesList",
        "desc": "최근일자 전체가격. '오늘','현재'등 최신전체스냅샷",
        "fields": {},
    },
    "recent_price_trend": {
        "action": "recentlyPriceTrendList",
        "desc": "최근가격동향. '최근'등 모호한기간시. 필수:품목코드",
        "fields": {"p_regday": "기준일자", "p_productno": "품목코드(필수)"},
    },
    "monthly_price_trend": {
        "action": "monthlyPriceTrendList",
        "desc": "월평균 동향. 월별트렌드분석",
        "fields": {"p_productno": "품목코드", "p_regday": "기준일자"},
    },
    "yearly_price_trend": {
        "action": "yearlyPriceTrendList",
        "desc": "연평균 동향. 장기트렌드분석",
        "fields": {"p_productno": "품목코드", "p_regday": "기준일자"},
    },
    "daily_county": {
        "action": "dailyCountyList",
        "desc": "지역별 최근가격. 특정지역 전체품목",
        "fields": {"p_countrycode": "시군구코드"},
    },
    "old_eco_item": {
        "action": "NaturePriceList",
        "desc": "친환경가격(2005-2020.3). 특정일자",
        "fields": {
            "p_regday": "조회일자",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드",
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "new_eco_period": {
        "action": "periodEcoPriceList",
        "desc": "친환경가격(2020.4-현재). 기간별",
        "fields": {
            "p_startday": "시작일",
            "p_endday": "종료일",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드",
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "new_eco_item": {
        "action": "EcoPriceList",
        "desc": "친환경가격(2020.4-현재). 특정일자",
        "fields": {
            "p_regday": "조회일자",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드",
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "region_item": {
        "action": "ItemInfo",
        "desc": "지역별품목별가격. 특정품목+지역 조합시. 지역비교는 반복호출",
        "fields": {
            "p_productclscode": "구분(01=소매,02=도매)",
            "p_regday": "조회일자",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드(필수)",
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_countrycode": "시군구코드(필수)",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "product_info": {
        "action": "productInfo",
        "desc": "코드표조회. search_item실패시 사용",
        "fields": {
            "p_startday": "시작일",
            "p_endday": "종료일",
            "p_countrycode": "시군구코드",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드",
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "period_wholesale": {
        "action": "periodWholesaleProductList",
        "desc": "일별도매가격(기간). 도매전용",
        "fields": {
            "p_startday": "시작일",
            "p_endday": "종료일",
            "p_countrycode": "시군구코드",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드(필수)",
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "period_retail": {
        "action": "periodRetailProductList",
        "desc": "일별소매가격(기간). 소매전용",
        "fields": {
            "p_startday": "시작일",
            "p_endday": "종료일",
            "p_countrycode": "시군구코드",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": "품목코드(필수)",
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
}


# =========================
# 툴 생성
# =========================
def validate_dates(params: dict) -> Optional[str]:
    """날짜 검증"""
    for k in ("p_regday", "p_startday", "p_endday"):
        if params.get(k):
            try:
                dt.datetime.strptime(params[k], "%Y-%m-%d")
            except:
                return f"{k}는 YYYY-MM-DD 형식이어야 합니다"
    if params.get("p_startday") and params.get("p_endday"):
        if params["p_startday"] > params["p_endday"]:
            return "시작일이 종료일보다 늦습니다"
    return None


def make_api_tool(name: str, spec: dict, client: KamisClient) -> StructuredTool:
    """API를 Tool로 변환"""
    fields = {
        k: (Optional[str], Field(default=None, description=v))
        for k, v in spec["fields"].items()
    }
    InputModel = create_model(f"{name}_Input", **fields)

    def run(**kwargs):
        params = {k: v for k, v in kwargs.items() if v}
        if err := validate_dates(params):
            return {"error": err}
        data = client.call(spec["action"], params)
        return {"action": spec["action"], "params": params, "response": data}

    return StructuredTool.from_function(
        name=name,
        func=run,
        args_schema=InputModel,
        description=f"{spec['desc']}. 파라미터: {', '.join(spec['fields'].keys()) or '없음'}",
    )


def make_search_tool(searcher: LLMHierarchicalSearcher) -> StructuredTool:
    """품목 검색 툴"""

    class Input(BaseModel):
        natural_query: str = Field(description="품목명 또는 자연어")
        top_k: int = Field(3, ge=1, le=10, description="결과 개수")

    def run(natural_query: str, top_k: int = 3):
        if not natural_query.strip():
            return {"error": "검색어 필요"}

        try:
            results = searcher.search_hierarchical(natural_query.strip())
        except Exception as e:
            return {"error": str(e)}

        if not results:
            return {"candidates": [], "note": "결과없음"}

        candidates = []
        seen = set()
        for item in results[:top_k]:
            prod = item.get("product", {})
            code = prod.get("product_code")
            name = prod.get("product_name")
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
        func=run,
        args_schema=Input,
        description="품목명→코드변환. 품목명 나오면 필수호출. 예:사과,배추,돼지고기,고등어",
    )


# =========================
# LangGraph
# =========================
class AgentState(TypedDict):
    messages: List[Any]


def build_graph(llm: ChatOpenAI, tools: List[StructuredTool]):
    tool_node = ToolNode(tools)

    system_prompt = """KAMIS 농축수산물 가격정보 에이전트

## 의사결정 3단계 프로세스

[1단계] 품목코드 획득
- 품목명 인식 시 → search_item 필수 호출
- 예: "사과", "배추", "돼지고기", "고등어" 등
- product_code를 p_itemcode로 사용

[2단계] 시간범위로 API 선택
- 오늘/현재 → daily_sales_list(전체) 또는 region_item(특정품목)
- 기간 명시(~1년) → daily_by_item_period
- 기간 모호("최근") → recent_price_trend  
- 월 단위 → monthly_sales 또는 monthly_price_trend
- 1년 초과 → yearly_sales (필수!)

[3단계] 조회범위 결정
- 부류 전체 → daily_by_category (부류코드: 100=식량,200=채소,300=과일,400=특용,500=축산,600=수산)
- 특정 품목 → daily_by_item_period, region_item
- 지역 비교 → region_item 반복 호출 (각 지역별)

## 핵심 규칙
- 날짜: YYYY-MM-DD 형식
- 구분: 01=소매(기본), 02=도매
- 친환경: 2020.3 이전=old_eco, 2020.4 이후=new_eco
- 지역비교: 단일 API로 불가, 각 지역별 호출 필요

## 실전 예시

예1) "오늘 사과 가격?"
→ search_item("사과") → product_code=245
→ region_item(p_itemcode=245, p_regday=2025-10-13, p_productclscode=01)

예2) "지난 3개월 배추 추이"  
→ search_item("배추") → product_code=211
→ daily_by_item_period(p_itemcode=211, p_startday=2025-07-13, p_endday=2025-10-13)

예3) "서울 부산 고등어 비교"
→ search_item("고등어") → product_code=604
→ region_item(p_itemcode=604, p_countrycode=1101) # 서울
→ region_item(p_itemcode=604, p_countrycode=2601) # 부산

예4) "최근 2년 쌀 가격"
→ search_item("쌀") → product_code=111
→ yearly_sales(p_itemcode=111, p_yyyy=2025, p_period=2) # 1년 초과이므로!

## API 선택 주의사항
- daily_by_item_period vs recent_price_trend: 기간 명시 여부로 구분
- period_wholesale vs period_retail: 도매/소매 전용, daily_by_item_period는 둘다 가능
- monthly_sales vs monthly_price_trend: 전자는 집계, 후자는 트렌드 분석
- 1년 초과 기간: 반드시 yearly 계열 사용

## 출력 형식
생각: (왜 이 API를 선택했는지)
행동: (도구명)
행동입력: (핵심 파라미터)
관찰: (결과 요약)
최종답변: (가격, 단위, 날짜, 전일대비 등 구체적으로)

현재: 2025-10-13 (월요일)"""

    def call_model(state: AgentState):
        msgs = [SystemMessage(content=system_prompt)] + state["messages"]
        return {"messages": state["messages"] + [llm.bind_tools(tools).invoke(msgs)]}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "end"

    def call_tools(state: AgentState):
        return {"messages": state["messages"] + tool_node.invoke(state)["messages"]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")

    return graph.compile()


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(description="KAMIS Multi-Tool LLM Agent")
    parser.add_argument("--query", required=True, help="질의")
    parser.add_argument("--model", default="gpt-5-mini", help="모델")
    args = parser.parse_args()

    # 검증
    if not OPENAI_API_KEY:
        print("[오류] OPENAI_API_KEY 설정 필요", file=sys.stderr)
        sys.exit(1)

    if not KAMIS_CERT_KEY or not KAMIS_CERT_ID:
        print("[경고] KAMIS 인증키 미설정. www.kamis.or.kr에서 발급", file=sys.stderr)

    # 초기화
    try:
        client = KamisClient(KAMIS_CERT_KEY or "111", KAMIS_CERT_ID or "222")
        searcher = LLMHierarchicalSearcher(DB_PATH)

        tools = [make_search_tool(searcher)]
        tools.extend(
            make_api_tool(name, spec, client) for name, spec in API_DEFS.items()
        )

        llm = ChatOpenAI(model=args.model, temperature=0, reasoning_effort="minimal")
        app = build_graph(llm, tools)

    except FileNotFoundError:
        print(f"[오류] DB파일 없음: {DB_PATH}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[오류] 초기화 실패: {e}", file=sys.stderr)
        sys.exit(1)

    # 실행
    try:
        result = app.invoke({"messages": [HumanMessage(content=args.query)]})
        final = next(
            (m for m in reversed(result["messages"]) if isinstance(m, AIMessage)), None
        )

        if final:
            print(
                final.content
                if isinstance(final.content, str)
                else json.dumps(final.content, ensure_ascii=False, indent=2)
            )
        else:
            print("답변 생성 실패")

    except Exception as e:
        print(f"[오류] 실행 실패: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
