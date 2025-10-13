# -*- coding: utf-8 -*-
"""KAMIS Service 설정 관리"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
from dotenv import load_dotenv

from .exceptions import ConfigError


@dataclass
class Config:
    """서비스 설정"""

    # API 인증
    kamis_cert_key: str
    kamis_cert_id: str

    # OpenAI (GPT-5 계열)
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-5-mini"
    reasoning_effort: Literal["low", "medium", "high", "minimal"] = "minimal"

    # 데이터베이스
    db_path: str = "kamis_api_list.db"
    db_max_age_hours: int = 24

    # 다운로드
    excel_url: str = (
        "https://www.kamis.or.kr/customer/board/board_file.do"
        "?brdno=4&brdctsno=424245&brdctsfileno=15636"
    )

    @classmethod
    def from_env(
        cls,
        cert_key: Optional[str] = None,
        cert_id: Optional[str] = None,
        db_path: Optional[str] = None,
    ) -> "Config":
        """환경변수에서 설정 로드"""
        load_dotenv()

        # 우선순위: 파라미터 > 환경변수
        kamis_cert_key = cert_key or os.getenv("KAMIS_CERT_KEY")
        kamis_cert_id = cert_id or os.getenv("KAMIS_CERT_ID")

        if not kamis_cert_key or not kamis_cert_id:
            raise ConfigError(
                "KAMIS API 인증 정보가 필요합니다. "
                "환경변수(KAMIS_CERT_KEY, KAMIS_CERT_ID)를 설정하거나 "
                "파라미터로 전달하세요."
            )

        return cls(
            kamis_cert_key=kamis_cert_key,
            kamis_cert_id=kamis_cert_id,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            reasoning_effort=os.getenv("REASONING_EFFORT", "minimal"),
            db_path=db_path or os.getenv("DB_PATH", "kamis_api_list.db"),
            db_max_age_hours=int(os.getenv("DB_MAX_AGE_HOURS", "24")),
        )

    def validate(self) -> None:
        """설정 검증"""
        if not self.kamis_cert_key or not self.kamis_cert_id:
            raise ConfigError("KAMIS API 인증 정보가 유효하지 않습니다.")

        # DB 경로 디렉토리 존재 확인
        db_dir = Path(self.db_path).parent
        if not db_dir.exists():
            db_dir.mkdir(parents=True, exist_ok=True)

        # reasoning_effort 값 검증
        valid_efforts = ["low", "medium", "high", "minimal"]
        if self.reasoning_effort not in valid_efforts:
            raise ConfigError(f"reasoning_effort는 {valid_efforts} 중 하나여야 합니다.")
