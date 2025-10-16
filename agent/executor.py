# -*- coding: utf-8 -*-
"""LangGraph Agent 실행기"""

import logging
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from ..core.client import KamisClient
from ..search import HierarchicalSearcher
from .tool_factory import ToolFactory
from .prompts import get_system_prompt

logger = logging.getLogger(__name__)


class KamisAgent:
    """KAMIS Agent (LangGraph create_react_agent 기반)"""

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

        # LangGraph Agent 구성 (create_react_agent 사용)
        self.graph = self._build_graph()

        logger.info(f"Agent 초기화 완료: {len(self.tools)}개 Tool")

    def _build_graph(self):
        """LangGraph Agent 구성 (create_react_agent 사용)"""
        return create_react_agent(
            self.llm,
            self.tools,
            prompt=SystemMessage(content=get_system_prompt()),
        )

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

            # 마지막 AIMessage 찾기 (Tool 호출이 아닌)
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
