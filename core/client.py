# -*- coding: utf-8 -*-
"""KAMIS API 클라이언트"""

import requests
from typing import Any, Dict, Optional
from ..exceptions import APIError


class KamisClient:
    """KAMIS Open API 클라이언트"""

    BASE_URL = "http://www.kamis.or.kr/service/price/xml.do"
    TIMEOUT = 20

    def __init__(self, cert_key: str, cert_id: str):
        self.cert_key = cert_key
        self.cert_id = cert_id
        self.session = requests.Session()

    def call(
        self, action: str, params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        API 호출

        Args:
            action: API 액션명 (예: dailySalesList)
            params: 추가 파라미터

        Returns:
            API 응답 (JSON)

        Raises:
            APIError: API 호출 실패 시
        """
        query_params = {
            "action": action,
            "p_cert_key": self.cert_key,
            "p_cert_id": self.cert_id,
            "p_returntype": "json",
        }

        if params:
            query_params.update(params)

        try:
            response = self.session.get(
                self.BASE_URL, params=query_params, timeout=self.TIMEOUT
            )
            response.raise_for_status()

            if not response.text:
                return {"error": "빈 응답"}

            return response.json()

        except requests.Timeout:
            raise APIError("API 타임아웃", status_code=408)
        except requests.HTTPError as e:
            raise APIError(
                f"API 호출 실패: {e.response.status_code}",
                status_code=e.response.status_code,
            )
        except requests.RequestException as e:
            raise APIError(f"네트워크 오류: {str(e)}")
        except Exception as e:
            raise APIError(f"알 수 없는 오류: {str(e)}")

    def close(self) -> None:
        """세션 종료"""
        self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
