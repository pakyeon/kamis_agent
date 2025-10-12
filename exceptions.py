# -*- coding: utf-8 -*-
"""KAMIS Service 예외 클래스"""

from typing import Optional


class KamisError(Exception):
    """기본 예외 클래스"""

    pass


class ConfigError(KamisError):
    """설정 관련 예외"""

    pass


class DatabaseError(KamisError):
    """데이터베이스 관련 예외"""

    pass


class APIError(KamisError):
    """API 호출 관련 예외"""

    def __init__(self, message: str, status_code: Optional[int] = None):
        super().__init__(message)
        self.status_code = status_code


class ItemNotFoundError(KamisError):
    """품목을 찾을 수 없음"""

    pass


class ValidationError(KamisError):
    """입력 검증 실패"""

    pass
