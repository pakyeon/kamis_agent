# -*- coding: utf-8 -*-
"""LangGraph Agent 실행기"""

import logging
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict

from ..core.client import KamisClient
from ..search import HierarchicalSearcher
from .tool_factory import ToolFactory
from .prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """Agent 상태"""

    messages: List[Any]


class KamisAgent:
    """KAMIS Agent (LangGraph 기반)"""

    def __init__(
        self,
        client: KamisClient,
        searcher: HierarchicalSearcher,
        llm: ChatOpenAI,
    ):
        self.client = client
        self.searcher = searcher
        self.llm = llm

        # Tool 생성
        tool_factory = ToolFactory(client, searcher)
        self.tools = tool_factory.create_all_tools()

        # LangGraph 구성
        self.graph = self._build_graph()

        logger.info(f"Agent 초기화 완료: {len(self.tools)}개 Tool")

    def _build_graph(self):
        """LangGraph 구성"""

        # Tool Node
        tool_node = ToolNode(self.tools)

        # Agent Node
        def call_model(state: AgentState):
            """LLM 호출"""
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
            response = self.llm.bind_tools(self.tools).invoke(messages)
            return {"messages": state["messages"] + [response]}

        # 조건부 엣지
        def should_continue(state: AgentState):
            """다음 노드 결정"""
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                return "tools"
            return "end"

        # Tool Node
        def call_tools(state: AgentState):
            """Tool 실행"""
            return {"messages": state["messages"] + tool_node.invoke(state)["messages"]}

        # Graph 구성
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", call_tools)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent", should_continue, {"tools": "tools", "end": END}
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def execute(self, query: str) -> Dict[str, Any]:
        """
        자연어 쿼리 실행

        Args:
            query: 자연어 질문 (예: "오늘 사과 가격은?")

        Returns:
            실행 결과
        """
        try:
            # Graph 실행
            result = self.graph.invoke({"messages": [HumanMessage(content=query)]})

            # 최종 메시지 추출
            messages = result.get("messages", [])
            final_message = None

            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    final_message = msg
                    break

            if final_message:
                return {
                    "success": True,
                    "answer": final_message.content,
                    "message_count": len(messages),
                }
            else:
                return {"success": False, "error": "답변 생성 실패"}

        except Exception as e:
            logger.error(f"Agent 실행 실패: {e}")
            return {"success": False, "error": str(e)}

    def get_tool_names(self) -> List[str]:
        """Tool 목록 반환"""
        return [tool.name for tool in self.tools]
