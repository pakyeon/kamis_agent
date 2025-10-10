#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
KAMIS 멀티-툴 LLM 에이전트 (CLI)
- LangGraph 중심, LangChain 툴 기반
- 17개 Open API 각각을 Tool로 래핑
- 품목명→코드 매핑: productInfo 캐시 + search_item 툴 제공
- 출력 포맷: ReAct 로그(간단) + 최종답변
"""

import os
import sys
import json
import argparse
import datetime as dt
from typing import Any, Dict, List, Optional, TypedDict

import requests
from requests.exceptions import Timeout, HTTPError, RequestException
from pydantic import BaseModel, Field, create_model
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# 품목 검색 모듈
from llm_product_searcher import LLMProductSearcher

load_dotenv()

# =========================
# 환경변수 (보안: 하드코딩 금지)
# =========================
KAMIS_CERT_KEY = os.getenv("KAMIS_CERT_KEY")
KAMIS_CERT_ID = os.getenv("KAMIS_CERT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DB_PATH = os.getenv("DB_PATH", "kamis_api_list.db")

if not KAMIS_CERT_KEY or not KAMIS_CERT_ID:
    print(
        "[경고] KAMIS 인증 env(KAMIS_CERT_KEY, KAMIS_CERT_ID)가 설정되지 않았습니다. "
        "테스트라면 111/222를 환경변수로 설정하세요.",
        file=sys.stderr,
    )


# =========================
# KAMIS 클라이언트
# =========================
class KamisClient:
    BASE_URL = "http://www.kamis.or.kr/service/price/xml.do"
    TIMEOUT = 20

    def __init__(
        self, cert_key: str, cert_id: str, session: Optional[requests.Session] = None
    ):
        self.cert_key = cert_key
        self.cert_id = cert_id
        self.sess = session or requests.Session()

    def call(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """KAMIS API 호출(p_returntype=json)."""
        q = dict(params or {})
        q["action"] = action
        q["p_cert_key"] = self.cert_key
        q["p_cert_id"] = self.cert_id
        q["p_returntype"] = "json"
        try:
            resp = self.sess.get(self.BASE_URL, params=q, timeout=self.TIMEOUT)
            resp.raise_for_status()
        except Timeout:
            return {"error": "API 타임아웃(20초 초과)", "params": q}
        except HTTPError as e:
            status = getattr(e.response, "status_code", None)
            body = getattr(e.response, "text", "") or ""
            return {"error": f"HTTP 에러 {status}", "detail": body[:500], "params": q}
        except RequestException as e:
            return {"error": "요청 오류", "detail": str(e), "params": q}

        # 일부 응답이 JSON이 아닐 수 있어 방어
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text}

        if not data:
            return {"error": "응답 없음", "params": q}
        return data


# =========================
# 제품/등급 코드북 (#15 productInfo)
# =========================
class CodeBook:
    def __init__(self, client: KamisClient):
        self.client = client
        self.cache = None  # 전체 코드표 캐시

    def _fetch(self):
        data = self.client.call("productInfo", {})
        self.cache = data

    def ensure_loaded(self):
        if self.cache is None:
            self._fetch()


# =========================
# 툴 스키마/검증 헬퍼
# =========================
def date_str(v: str) -> str:
    """YYYY-MM-DD 형식 검증/정규화."""
    try:
        return dt.datetime.strptime(v, "%Y-%m-%d").strftime("%Y-%m-%d")
    except Exception:
        raise ValueError("날짜는 YYYY-MM-DD 형식이어야 합니다.")


def _validate_dates(q: Dict[str, Any]) -> Optional[str]:
    """요청 파라미터의 날짜 관련 기본 검증."""
    for k in ("p_regday", "p_startday", "p_endday"):
        if q.get(k):
            try:
                q[k] = date_str(q[k])
            except ValueError as e:
                return str(e)
    if q.get("p_startday") and q.get("p_endday"):
        if q["p_startday"] > q["p_endday"]:
            return "시작일이 종료일보다 늦습니다."
    return None


def _summarize_kamis_response(action: str, data: Dict[str, Any]) -> str:
    """LLM이 해석하기 쉽도록 몇 가지 핵심값을 간단 요약."""
    try:
        # 간단 규칙: 일/기간 계열에서 날짜+가격 스냅샷 2~3개 추출
        def collect_rows(obj, rows):
            if isinstance(obj, dict):
                # 흔한 키들
                cand_date = (
                    obj.get("date")
                    or obj.get("regday")
                    or obj.get("ymd")
                    or obj.get("day")
                )
                cand_price = (
                    obj.get("dpr1")
                    or obj.get("price")
                    or obj.get("avg_price")
                    or obj.get("value")
                    or obj.get("amt")
                )
                if cand_date and cand_price:
                    unit = obj.get("unit") or obj.get("unit_qty") or ""
                    rows.append((str(cand_date), str(cand_price), str(unit)))
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        collect_rows(v, rows)
            elif isinstance(obj, list):
                for it in obj:
                    collect_rows(it, rows)

        rows: List[tuple] = []
        collect_rows(data, rows)
        if rows:
            head = rows[:3]
            return " | ".join(
                [
                    f"{d}: {p}{(' ' + u) if u and u != 'None' else ''}"
                    for d, p, u in head
                ]
            )

        # 보조: item_name, kind_name, dpr1 등 3~6개 키 스냅샷
        def find_keys(obj, keys, found):
            if len(found) >= 6:
                return
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k in keys and len(found) < 6:
                        found.append((k, v))
                    if isinstance(v, (dict, list)):
                        find_keys(v, keys, found)
            elif isinstance(obj, list):
                for it in obj:
                    find_keys(it, keys, found)

        found = []
        find_keys(data, {"item_name", "kind_name", "dpr1", "dpr3", "unit"}, found)
        if found:
            return ", ".join([f"{k}={v}" for k, v in found[:6]])
    except Exception:
        pass
    return ""


# =========================
# 각 API별 정의
# =========================
API_DEFS: Dict[str, Dict[str, Any]] = {
    # 1. 일별 부류별 도/소매
    "daily_by_category": {
        "action": "dailyPriceByCategoryList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_product_cls_code": (str, "구분(01:소매, 02:도매)"),
            "p_item_category_code": (str, "부류코드(100~600)"),
            "p_country_code": (str, "시군구코드"),
            "p_regday": (str, "조회일자 YYYY-MM-DD"),
            "p_convert_kg_yn": (str, "kg단위 환산여부(Y/N)"),
        },
    },
    # 2. 일별 품목별 도/소매 (기간형, 최대 1년)
    "daily_by_item_period": {
        "action": "periodProductList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_startday": (str, "조회 시작일 YYYY-MM-DD"),
            "p_endday": (str, "조회 종료일 YYYY-MM-DD"),
            "p_productclscode": (str, "구분(01:소매, 02:도매)"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_itemcode": (str, "품목코드"),
            "p_kindcode": (str, "품종코드"),
            "p_productrankcode": (str, "등급코드"),
            "p_countrycode": (str, "시군구코드"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
        },
    },
    # 3. 월별 도/소매
    "monthly_sales": {
        "action": "monthlySalesList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_yyyy": (str, "연도(YYYY)"),
            "p_period": (str, "기간(년수), default:3"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_kindcode": (str, "품종코드"),
            "p_graderank": (str, "등급(상품:1,중품:2 등)"),
            "p_countrycode": (str, "시군구코드"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
            "p_itemcode": (str, "품목코드"),
        },
    },
    # 4. 연도별 도/소매
    "yearly_sales": {
        "action": "yearlySalesList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_kindcode": (str, "품종코드"),
            "p_graderank": (str, "등급(상품:1,중품:2 등)"),
            "p_countrycode": (str, "시군구코드"),
            "p_itemcode": (str, "품목코드"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_yyyy": (str, "연도(YYYY)"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
        },
    },
    # 5. (구) 친환경 (’05~’20.3.) - 기간
    "old_eco_period": {
        "action": "periodNaturePriceList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_startday": (str, "조회 시작일 YYYY-MM-DD"),
            "p_endday": (str, "조회 종료일 YYYY-MM-DD"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_itemcode": (str, "품목코드"),
            "p_kindcode": (str, "품종코드"),
            "p_productrankcode": (str, "등급코드"),
            "p_countrycode": (str, "시군구코드"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
        },
    },
    # 6. 최근일자 도/소매 (상품 기준)
    "daily_sales_list": {
        "action": "dailySalesList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
        },
    },
    # 7. 최근 가격추이 (상품 기준)
    "recent_price_trend": {
        "action": "recentlyPriceTrendList",
        "fields": {
            "p_regday": (str, "검색일자 YYYY-MM-DD"),
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_productno": (str, "품목코드(필수)"),
        },
    },
    # 8. 월평균 가격추이
    "monthly_price_trend": {
        "action": "monthlyPriceTrendList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_productno": (str, "품목코드"),
            "p_regday": (str, "검색일자 YYYY-MM-DD"),
        },
    },
    # 9. 연평균 가격추이
    "yearly_price_trend": {
        "action": "yearlyPriceTrendList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_productno": (str, "품목코드"),
            "p_regday": (str, "검색일자 YYYY-MM-DD"),
        },
    },
    # 10. 최근일자 지역별 도/소매
    "daily_county": {
        "action": "dailyCountyList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_countrycode": (str, "시군구코드"),
        },
    },
    # 11. (구) 친환경 품목별 (’05~’20.3.)
    "old_eco_item": {
        "action": "NaturePriceList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_regday": (str, "조회일자 YYYY-MM-DD"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_itemcode": (str, "품목코드"),
            "p_kindcode": (str, "품종코드"),
            "p_productrankcode": (str, "등급코드"),
            "p_countrycode": (str, "시군구코드"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
        },
    },
    # 12. (신) 친환경 (’20.4~) - 기간
    "new_eco_period": {
        "action": "periodEcoPriceList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_startday": (str, "조회 시작일 YYYY-MM-DD"),
            "p_endday": (str, "조회 종료일 YYYY-MM-DD"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_itemcode": (str, "품목코드"),
            "p_kindcode": (str, "품종코드"),
            "p_productrankcode": (str, "등급코드"),
            "p_countrycode": (str, "시군구코드"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
        },
    },
    # 13. (신) 친환경 품목별 (’20.4~)
    "new_eco_item": {
        "action": "EcoPriceList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_regday": (str, "조회일자 YYYY-MM-DD"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_itemcode": (str, "품목코드"),
            "p_kindcode": (str, "품종코드"),
            "p_productrankcode": (str, "등급코드"),
            "p_countrycode": (str, "시군구코드"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
        },
    },
    # 14. 지역별 품목별 도/소매
    "region_item": {
        "action": "ItemInfo",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_productclscode": (str, "구분(01:소매, 02:도매)"),
            "p_regday": (str, "조회일자 YYYY-MM-DD"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_itemcode": (str, "품목코드"),
            "p_kindcode": (str, "품종코드"),
            "p_productrankcode": (str, "등급코드"),
            "p_countrycode": (str, "시군구코드"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
        },
    },
    # 15. 품목 및 등급 코드표
    "product_info": {
        "action": "productInfo",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_startday": (str, "조회 시작일 YYYY-MM-DD"),
            "p_countrycode": (str, "시군구코드"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_itemcode": (str, "품목코드"),
            "p_kindcode": (str, "품종코드"),
            "p_productrankcode": (str, "등급코드"),
            "p_endday": (str, "조회 종료일 YYYY-MM-DD"),
        },
    },
    # 16. 신) 일별 품목별 도매 (기간)
    "period_wholesale": {
        "action": "periodWholesaleProductList",
        "fields": {
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_startday": (str, "조회 시작일 YYYY-MM-DD"),
            "p_endday": (str, "조회 종료일 YYYY-MM-DD"),
            "p_countrycode": (str, "시군구코드"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_itemcode": (str, "품목코드"),
            "p_kindcode": (str, "품종코드"),
            "p_productrankcode": (str, "등급코드"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
        },
    },
    # 17. 신) 일별 품목별 소매 (기간)
    "period_retail": {
        "action": "periodRetailProductList",
        "fields": {
            "p_countrycode": (str, "시군구코드"),
            "p_returntype": (str, "반환 형식(json, xml)"),
            "p_startday": (str, "조회 시작일 YYYY-MM-DD"),
            "p_productrankcode": (str, "등급코드"),
            "p_convert_kg_yn": (str, "kg 환산 Y/N"),
            "p_itemcategorycode": (str, "부류코드"),
            "p_itemcode": (str, "품목코드"),
            "p_kindcode": (str, "품종코드"),
            "p_endday": (str, "조회 종료일 YYYY-MM-DD"),
        },
    },
}


# =========================
# 동적 Tool 생성 (+에러/날짜검증/요약)
# =========================
def make_structured_tool(
    name: str, action: str, fields: Dict[str, Any], client: KamisClient
) -> StructuredTool:
    """API_DEFS를 LangChain StructuredTool로 변환."""
    field_definitions = {
        k: (Optional[tp], Field(default=None, description=desc))
        for k, (tp, desc) in fields.items()
    }
    model = create_model(f"{name}_InputModel", **field_definitions)

    def _run(**kwargs):
        # None 값 제거
        q = {k: v for k, v in kwargs.items() if v is not None}
        # 날짜 검증
        err = _validate_dates(q)
        if err:
            return {"error": err, "action": action, "params": q}

        data = client.call(action, q)
        # client.call에서 에러 dict를 반환했을 수 있음
        if isinstance(data, dict) and data.get("error"):
            return {
                "action": action,
                "params": q,
                "error": data.get("error"),
                "detail": data.get("detail"),
            }

        summary = (
            _summarize_kamis_response(action, data) if isinstance(data, dict) else ""
        )
        return {"action": action, "params": q, "summary": summary, "response": data}

    return StructuredTool.from_function(
        name=name,
        func=lambda **kw: _run(**kw),
        args_schema=model,
        description=f"KAMIS OpenAPI Tool: action={action} (파라미터: {', '.join(list(fields.keys()) or ['(없음)'])})",
    )


# =========================
# search_item 툴 (CodeBook 실제 활용)
# =========================
def make_search_item_tool(
    codebook: CodeBook, searcher: Optional[LLMProductSearcher] = None
) -> StructuredTool:
    class _Input(BaseModel):
        natural_query: Optional[str] = Field(None, description="자연어 질의")
        category_hint: Optional[str] = Field(
            None, description="카테고리 힌트 (예: 과일, 채소)"
        )
        top_k: int = Field(3, ge=1, le=10, description="후보 개수(1~10, 기본 3)")

    def _run(
        natural_query: Optional[str] = None,
        category_hint: Optional[str] = None,
        top_k: int = 3,
    ):
        query_text = (natural_query or "").strip()
        if not query_text:
            return {"error": "검색어가 비어 있습니다. natural_query를 지정하세요."}

        # 1) 우선 LLM+FTS5 기반 검색 시도
        pairs = searcher.get_name_code_pairs(
            query_text
        )  # [{product_name, product_code}, ...]
        cands = []
        for p in pairs[: max(1, min(top_k, 10))]:
            name = p.get("product_name")
            code = p.get("product_code")
            cands.append(
                {
                    "product_name": name,
                    "product_code": code,
                    # 하위 호환 필드(기존 파이프라인에서 기대할 수 있음)
                    "item_name": name,
                    "itemcode": code,
                    "source": "llm_fts",
                }
            )
        if cands:
            return {
                "query": {
                    "natural": query_text,
                    "category_hint": category_hint,
                },
                "candidates": cands,
                "note": "LLM+FTS5 기반 고급 검색 결과입니다. 필요 시 product_info로 품종/등급 코드 보완.",
            }

    return StructuredTool.from_function(
        name="search_item",
        func=_run,
        args_schema=_Input,
        description=(
            "품목명/자연어 질의로 KAMIS 코드 후보를 검색합니다." "반환: 후보 상위 N개."
        ),
    )


# =========================
# LangGraph 상태/노드 정의
# =========================
class AgentState(TypedDict):
    messages: List[Any]
    trace: List[Dict[str, Any]]  # 간단 ReAct 로그 저장용


def make_tools(
    client: KamisClient, codebook: CodeBook, searcher: Optional[LLMProductSearcher]
) -> List[StructuredTool]:
    tools: List[StructuredTool] = []
    tools.append(make_search_item_tool(codebook, searcher))  # ← 검색기 주입
    for tool_name, spec in API_DEFS.items():
        tools.append(
            make_structured_tool(
                name=tool_name,
                action=spec["action"],
                fields=spec["fields"],
                client=client,
            )
        )
    return tools


def build_graph(llm: ChatOpenAI, tools: List[StructuredTool]):
    tool_node = ToolNode(tools)

    def call_model(state: AgentState):
        """메인 에이전트: 도구 선택/호출 계획."""
        system = (
            "당신은 KAMIS 농축수산물 가격 정보 전문 에이전트입니다.\n"
            "요청을 분석하여 필요하면 먼저 'search_item' 또는 'product_info'로 품목코드를 확인하고, 적절한 도구를 호출해 결과를 한국어로 요약하세요.\n"
            "출력 형식:\n"
            "생각: (간단 계획 1줄)\n"
            "행동: (사용한 도구명)\n"
            "행동입력: (핵심 파라미터 요약)\n"
            "관찰: (핵심 수치/포인트 1~2줄; 툴 응답의 summary를 활용)\n"
            "...\n"
            "최종답변: (문장형 한국어 요약; 단위/기간/지역/구분 명확히)\n"
        )
        msgs = [SystemMessage(content=system)] + list(state["messages"])
        ai = llm.bind_tools(tools).invoke(msgs)

        # 간단 trace 기록
        new_trace = list(state.get("trace", []))
        step = len(new_trace) + 1
        tool_names = (
            [tc.get("name") for tc in (ai.tool_calls or [])]
            if isinstance(ai, AIMessage)
            else []
        )
        thought_line = None
        if isinstance(ai.content, str):
            for line in ai.content.splitlines():
                if line.strip().startswith("생각"):
                    thought_line = line.strip()
                    break
        new_trace.append({"step": step, "tools": tool_names, "thought": thought_line})
        return {"messages": state["messages"] + [ai], "trace": new_trace}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "finalize"

    def record_tool_results(state: AgentState):
        """도구 실행 -> ToolNode."""
        out = tool_node.invoke(state)  # {"messages": [ToolMessage, ...]}
        tool_msgs = out.get("messages", [])
        # trace에 간단한 관찰 요약 포함(가능 시)
        new_trace = list(state.get("trace", []))
        for tm in tool_msgs:
            if isinstance(tm, ToolMessage):
                # 툴 응답에 summary가 있으면 trace에 반영
                try:
                    payload = (
                        json.loads(tm.content)
                        if isinstance(tm.content, str)
                        else tm.content
                    )
                except Exception:
                    payload = tm.content
                sm = None
                if isinstance(payload, dict):
                    sm = payload.get("summary")
                new_trace.append(
                    {
                        "step": len(new_trace) + 1,
                        "tool_result_for": tm.name,
                        "summary": sm,
                    }
                )
        return {"messages": state["messages"] + tool_msgs, "trace": new_trace}

    def finalize(state: AgentState):
        """최종 요약 단계(필요시 후처리 가능)."""
        return {"messages": state["messages"], "trace": state.get("trace", [])}

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", record_tool_results)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "finalize": "finalize"}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("finalize", END)
    return graph.compile()


# =========================
# CLI 실행
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="KAMIS 멀티-툴 LLM 에이전트 (LangGraph)"
    )
    parser.add_argument("--query", type=str, required=True, help="사용자 자연어 질의")
    parser.add_argument(
        "--model", type=str, default="gpt-5-mini", help="OpenAI 모델 (기본: gpt-5)"
    )
    parser.add_argument(
        "--trace", action="store_true", help="간단 trace 로그를 stderr로 출력"
    )
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("[오류] OPENAI_API_KEY 환경변수를 설정하세요.", file=sys.stderr)
        sys.exit(1)
    if not KAMIS_CERT_KEY or not KAMIS_CERT_ID:
        print(
            "[경고] KAMIS_CERT_KEY / KAMIS_CERT_ID가 비어 있습니다. 테스트값(111/222)을 환경변수로 설정하세요.",
            file=sys.stderr,
        )

    client = KamisClient(
        cert_key=KAMIS_CERT_KEY or "111", cert_id=KAMIS_CERT_ID or "222"
    )
    codebook = CodeBook(client)
    searcher = LLMProductSearcher(DB_PATH)

    tools = make_tools(client, codebook, searcher)  # ← 검색기 주입
    llm = ChatOpenAI(
        model=args.model,
        temperature=0,
        reasoning_effort="minimal",
        api_key=OPENAI_API_KEY,
    )

    app = build_graph(llm, tools)

    # LangGraph 실행
    state: AgentState = {"messages": [HumanMessage(content=args.query)], "trace": []}
    out = app.invoke(state)

    # trace(optional)
    if args.trace:
        print(
            "[TRACE]",
            json.dumps(out.get("trace", []), ensure_ascii=False, indent=2),
            file=sys.stderr,
        )

    # 최종 출력
    final_ai = None
    for m in out["messages"][::-1]:
        if isinstance(m, AIMessage):
            final_ai = m
            break
    if final_ai is None:
        print("최종답변: 죄송합니다. 답변을 생성하지 못했습니다.")
        return

    content = final_ai.content
    if isinstance(content, str):
        print(content)
    else:
        print(json.dumps(content, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
