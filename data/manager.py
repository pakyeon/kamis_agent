# -*- coding: utf-8 -*-
"""KAMIS 데이터 관리 (다운로드, 추출, 변환, 적재)"""

import os
import time
import tempfile
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from .downloader import ExcelDownloader
from .extractor import DataExtractor
from .transformer import DataTransformer
from .loader import DataLoader
from ..exceptions import DatabaseError

logger = logging.getLogger(__name__)


class DataManager:
    """KAMIS 데이터 전체 관리"""

    def __init__(
        self,
        db_path: str,
        excel_url: Optional[str] = None,
        table_name: str = "api_items",
    ):
        self.db_path = db_path
        self.table_name = table_name

        self.downloader = ExcelDownloader(excel_url)
        self.extractor = DataExtractor()
        self.transformer = DataTransformer()
        self.loader = DataLoader()

    def update_if_needed(self, max_age_hours: int = 24) -> bool:
        """
        DB가 오래되었으면 자동 업데이트

        Args:
            max_age_hours: DB 최대 유효 시간 (시간)

        Returns:
            업데이트 수행 여부
        """
        if not self._should_update(max_age_hours):
            logger.info(f"DB가 최신입니다 ({self.db_path})")
            return False

        logger.info("DB 업데이트 시작")
        self.update(auto_download=True)
        return True

    def update(
        self, excel_path: Optional[str] = None, auto_download: bool = True
    ) -> pd.DataFrame:
        """
        데이터 업데이트 (다운로드 → 추출 → 변환 → 적재)

        Args:
            excel_path: 엑셀 파일 경로 (없으면 자동 다운로드)
            auto_download: 자동 다운로드 여부

        Returns:
            최종 데이터프레임
        """
        temp_dir = None

        try:
            # 1. 엑셀 파일 준비
            if auto_download:
                temp_dir = tempfile.TemporaryDirectory(prefix="kamis_")
                excel_path = os.path.join(temp_dir.name, "kamis_codes.xlsx")
                self.downloader.download(excel_path)
            elif not excel_path or not os.path.exists(excel_path):
                raise DatabaseError(f"엑셀 파일을 찾을 수 없습니다: {excel_path}")

            # 2. 데이터 추출
            logger.info("데이터 추출 중...")
            sheets_data = self.extractor.extract_all(excel_path)

            # 3. 데이터 변환
            logger.info("데이터 변환 중...")
            final_data = self.transformer.transform(sheets_data)

            # 4. DB 적재
            logger.info("DB 적재 중...")
            self.loader.load(final_data, self.db_path, self.table_name)

            logger.info(f"업데이트 완료: {len(final_data)}행")
            return final_data

        finally:
            # 임시 파일 정리
            if temp_dir:
                try:
                    temp_dir.cleanup()
                except Exception as e:
                    logger.warning(f"임시 파일 정리 실패: {e}")

    def _should_update(self, max_age_hours: int) -> bool:
        """DB 업데이트 필요 여부 확인"""
        if not os.path.exists(self.db_path):
            logger.info("DB가 존재하지 않습니다")
            return True

        db_mtime = os.path.getmtime(self.db_path)
        age_hours = (time.time() - db_mtime) / 3600.0

        if age_hours > max_age_hours:
            logger.info(
                f"DB가 오래되었습니다: {age_hours:.1f}h (임계: {max_age_hours}h)"
            )
            return True

        return False
