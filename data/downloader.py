# -*- coding: utf-8 -*-
"""엑셀 파일 다운로더"""

import os
import logging
import requests
from typing import Optional, Dict

from ..exceptions import DatabaseError

logger = logging.getLogger(__name__)


class ExcelDownloader:
    """KAMIS 엑셀 파일 다운로드"""

    DEFAULT_URL = (
        "https://www.kamis.or.kr/customer/board/board_file.do"
        "?brdno=4&brdctsno=424245&brdctsfileno=15636"
    )

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.kamis.or.kr/customer/board/board.do",
    }

    TIMEOUT = 60

    def __init__(
        self, url: Optional[str] = None, headers: Optional[Dict[str, str]] = None
    ):
        self.url = url or self.DEFAULT_URL
        self.headers = headers or self.DEFAULT_HEADERS

    def download(self, dest_path: str) -> None:
        """
        파일 다운로드

        Args:
            dest_path: 저장 경로

        Raises:
            DatabaseError: 다운로드 실패 시
        """
        try:
            logger.info(f"다운로드 시작: {self.url}")

            response = requests.get(
                self.url, headers=self.headers, timeout=self.TIMEOUT
            )
            response.raise_for_status()

            # 디렉토리 생성
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)

            # 파일 저장
            with open(dest_path, "wb") as f:
                f.write(response.content)

            logger.info(f"다운로드 완료: {dest_path}")

        except requests.Timeout:
            raise DatabaseError("다운로드 타임아웃")
        except requests.RequestException as e:
            raise DatabaseError(f"다운로드 실패: {e}")
        except IOError as e:
            raise DatabaseError(f"파일 저장 실패: {e}")
