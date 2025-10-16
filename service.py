# -*- coding: utf-8 -*-
"""KAMIS 서비스 메인 클래스"""

import logging
from typing import Optional, Dict, Any

from langchain_openai import ChatOpenAI

from .config import Config
from .core.client import KamisClient
from .search import HierarchicalSearcher
from .data import DataManager
from .exceptions import KamisError, ConfigError

logger = logging.getLogger(__name__)


class KamisService:
    """
    KAMIS 농축수산물 가격 정보 서비스

    다른 Agent에서 사용할 수 있는 고수준 API 제공
    """

    def __init__(
        self,
        cert_key: Optional[str] = None,
        cert_id: Optional[str] = None,
        db_path: Optional[str] = None,
        auto_init: bool = True,
    ):
        """
        Args:
            cert_key: KAMIS API 인증키 (없으면 환경변수 사용)
            cert_id: KAMIS API 인증ID (없으면 환경변수 사용)
            db_path: SQLite DB 경로
            auto_init: 자동 초기화 여부
        """
        self.config = Config.from_env(cert_key, cert_id, db_path)
        self.config.validate()

        self._client: Optional[KamisClient] = None
        self._searcher: Optional[HierarchicalSearcher] = None
        self._data_manager: Optional[DataManager] = None
        self._agent: Optional[Any] = None  # KamisAgent

        if auto_init:
            self.initialize()

    def initialize(self) -> None:
        """서비스 초기화 (DB 체크, 리소스 로드)"""
        logger.info("서비스 초기화 시작")

        # 1. DB 준비
        self._ensure_db_ready()

        # 2. API 클라이언트
        self._client = KamisClient(
            self.config.kamis_cert_key, self.config.kamis_cert_id
        )

        # 3. 검색 엔진 (LLM 포함)
        llm = None
        if self.config.openai_api_key:
            try:
                llm = ChatOpenAI(
                    model=self.config.openai_model,
                    temperature=0,
                    reasoning_effort=self.config.reasoning_effort,
                    api_key=self.config.openai_api_key,
                )
                logger.info(f"LLM 초기화 완료: {self.config.openai_model}")
            except Exception as e:
                logger.warning(f"LLM 초기화 실패: {e}. 단순 검색 모드로 동작")

        self._searcher = HierarchicalSearcher(self.config.db_path, llm=llm)

        # 4. Agent (항상 활성화 - 단일 API 통합)
        if llm:
            try:
                from .agent import KamisAgent

                self._agent = KamisAgent(self._client, self._searcher, llm)
                logger.info("Agent 초기화 완료")
            except Exception as e:
                logger.warning(f"Agent 초기화 실패: {e}")
        else:
            logger.warning("LLM 없음. Agent 비활성화")

        logger.info("서비스 초기화 완료")

    # ===== 고수준 API (2개) =====

    def search(self, natural_query: str) -> Dict[str, Any]:
        """
        자연어 쿼리 → 검색 결과
        단순 검색: LLM으로 키워드 추출 → DB 검색 → JSON 반환

        Args:
            natural_query: 자연어 질문 (예: "사과 가격", "서울 부산 고등어 비교")

        Returns:
            부류/품목/품종/등급 정보를 포함한 Json 데이터
        """
        if not self._searcher:
            raise ConfigError(
                "검색 엔진이 초기화되지 않았습니다. " "KamisService()로 초기화하세요."
            )

        try:
            # 검색 엔진으로 직접 검색
            search_results = self._searcher.search(natural_query)

            return {
                "success": True,
                "query": natural_query,
                "items": search_results,
                "count": len(search_results),
            }

        except Exception as e:
            logger.error(f"검색 실패: {e}")
            return {
                "success": False,
                "error": {
                    "code": "SEARCH_FAILED",
                    "message": str(e),
                    "type": type(e).__name__,
                },
            }

    def answer(self, natural_query: str) -> str:
        """
        자연어 쿼리 → 검색 결과 → 자연어 답변

        Args:
            natural_query: 자연어 질문

        Returns:
            자연어 답변 (str)
        """
        if not self._agent:
            raise ConfigError(
                "Agent가 초기화되지 않았습니다. "
                "OPENAI_API_KEY를 설정하고 KamisService()로 초기화하세요."
            )

        try:
            # Agent 실행
            result = self._agent.execute(natural_query)

            if result.get("success"):
                return result.get("answer", "답변을 생성할 수 없습니다.")
            else:
                error_msg = result.get("error", "알 수 없는 오류")
                raise KamisError(f"답변 생성 실패: {error_msg}")

        except KamisError:
            raise
        except Exception as e:
            logger.error(f"답변 생성 중 오류: {e}")
            raise KamisError(f"답변 생성 실패: {e}") from e

    # ===== 내부 메서드 =====

    def resolve_query(
        self,
        query_text: str,
        top_k: int = 3,
    ) -> Optional[Dict[str, Any]]:
        """
        자연어 쿼리 해석

        Args:
            query_text: 자연어 쿼리
            top_k: 반환할 결과 개수

        Returns:
            구조화된 정보 (부류/품목/품종/등급/지역/시장 포함)
        """
        if not self._searcher:
            self.initialize()

        results = self._searcher.search(query_text, top_k=top_k)

        if not results:
            return None

        return results[0] if top_k == 1 else results

    # ===== 리소스 관리 =====

    def _ensure_db_ready(self) -> None:
        """DB 상태 확인 및 필요시 업데이트"""
        if not self._data_manager:
            self._data_manager = DataManager(
                self.config.db_path, excel_url=self.config.excel_url
            )

        self._data_manager.update_if_needed(max_age_hours=self.config.db_max_age_hours)

    def close(self) -> None:
        """리소스 정리"""
        if self._client:
            self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
