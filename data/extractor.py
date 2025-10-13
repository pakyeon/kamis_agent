# -*- coding: utf-8 -*-
"""엑셀 시트 데이터 추출"""

import logging
from typing import Dict, List, Optional
import pandas as pd

from ..exceptions import DatabaseError

logger = logging.getLogger(__name__)


class DataExtractor:
    """엑셀 시트별 데이터 추출"""

    # 시트별 필수 컬럼 정의
    SHEET_CONFIGS = {
        "부류코드": ["부류코드", "부류명"],
        "품목코드": ["부류코드", "품목코드", "품목명"],
        "품종코드": ["품목 코드", "품종코드", "품목명", "품종명"],
        "등급코드": [
            "등급코드(p_productrankcode)",
            "등급코드(p_graderank)",
            "등급코드명",
        ],
        "축산물 코드": [
            "품목명",
            "품종명",
            "등급명",
            "품목코드(itemcode)",
            "품종코드(kindcode)",
            "등급코드(periodProductList)",
        ],
        "산물코드": [
            "품목분류명",
            "품목분류코드",
            "품목명",
            "품목코드",
            "품종명",
            "품종코드",
            "산물등급명",
            "산물등급코드",
        ],
    }

    LIVESTOCK_CATEGORY_CODE = "500"
    LIVESTOCK_CATEGORY_NAME = "축산물"

    def extract_all(self, excel_path: str) -> Dict[str, pd.DataFrame]:
        """
        모든 시트 데이터 추출

        Returns:
            {"부류": df, "품목": df, ...}
        """
        logger.info(f"엑셀 읽기: {excel_path}")

        return {
            "부류": self._extract_category(excel_path),
            "품목": self._extract_product(excel_path),
            "품종": self._extract_kind(excel_path),
            "등급": self._extract_rank(excel_path),
            "축산물": self._extract_livestock(excel_path),
            "산물": self._extract_sanmul(excel_path),
        }

    def _read_sheet(
        self,
        excel_path: str,
        sheet_name: str,
        required_cols: List[str],
        max_scan: int = 50,
    ) -> pd.DataFrame:
        """시트 읽기 (헤더 자동 탐지)"""
        # 헤더 없이 읽기
        df_raw = pd.read_excel(
            excel_path, sheet_name=sheet_name, header=None, engine="openpyxl"
        )

        # 헤더 행 찾기
        header_row = self._find_header_row(df_raw, required_cols, max_scan)
        if header_row is None:
            raise DatabaseError(
                f"[{sheet_name}] 헤더를 찾을 수 없습니다: {required_cols}"
            )

        # 헤더를 기준으로 다시 읽기
        df = pd.read_excel(
            excel_path, sheet_name=sheet_name, header=header_row, engine="openpyxl"
        )
        df.columns = [str(c).strip() for c in df.columns]

        # 공백 제거 및 빈 행 제거
        df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
        df = df.replace("", pd.NA).dropna(how="all")

        # 필요한 컬럼만 선택
        for col in required_cols:
            if col not in df.columns:
                df[col] = pd.NA

        return df[required_cols]

    def _find_header_row(
        self, df_raw: pd.DataFrame, required_cols: List[str], max_scan: int
    ) -> Optional[int]:
        """헤더 행 찾기"""
        req_set = set(required_cols)

        for i in range(min(max_scan, len(df_raw))):
            headers = [
                str(v).strip() if pd.notna(v) else "" for v in df_raw.iloc[i].tolist()
            ]
            header_set = set(h for h in headers if h)

            if req_set.issubset(header_set):
                return i

        return None

    def _extract_category(self, excel_path: str) -> pd.DataFrame:
        """부류 시트 추출"""
        return self._read_sheet(excel_path, "부류코드", self.SHEET_CONFIGS["부류코드"])

    def _extract_product(self, excel_path: str) -> pd.DataFrame:
        """품목 시트 추출"""
        return self._read_sheet(excel_path, "품목코드", self.SHEET_CONFIGS["품목코드"])

    def _extract_kind(self, excel_path: str) -> pd.DataFrame:
        """품종 시트 추출"""
        df = self._read_sheet(excel_path, "품종코드", self.SHEET_CONFIGS["품종코드"])
        # 컬럼명 표준화
        df = df.rename(columns={"품목 코드": "품목코드"})
        return df[["품목코드", "품목명", "품종코드", "품종명"]]

    def _extract_rank(self, excel_path: str) -> pd.DataFrame:
        """등급 시트 추출"""
        return self._read_sheet(excel_path, "등급코드", self.SHEET_CONFIGS["등급코드"])

    def _extract_livestock(self, excel_path: str) -> pd.DataFrame:
        """축산물 시트 추출"""
        df = self._read_sheet(
            excel_path, "축산물 코드", self.SHEET_CONFIGS["축산물 코드"]
        )

        # 부류 정보 추가
        df["부류코드"] = self.LIVESTOCK_CATEGORY_CODE
        df["부류명"] = self.LIVESTOCK_CATEGORY_NAME

        # 컬럼명 표준화
        df = df.rename(
            columns={
                "품목코드(itemcode)": "품목코드",
                "품종코드(kindcode)": "축산_품종코드",
                "등급코드(periodProductList)": "등급코드(periodProductList)",
                "등급명": "등급코드(periodProductName)",
            }
        )

        return df[
            [
                "부류코드",
                "부류명",
                "품목코드",
                "품목명",
                "축산_품종코드",
                "품종명",
                "등급코드(periodProductList)",
                "등급코드(periodProductName)",
            ]
        ]

    def _extract_sanmul(self, excel_path: str) -> pd.DataFrame:
        """산물 시트 추출"""
        df = self._read_sheet(excel_path, "산물코드", self.SHEET_CONFIGS["산물코드"])

        # 컬럼명 표준화
        df = df.rename(
            columns={
                "품목분류코드": "부류코드",
                "품목분류명": "부류명",
                "산물등급명": "등급코드명",
                "산물등급코드": "등급코드(p_productrankcode)",
            }
        )

        return df[
            [
                "부류코드",
                "부류명",
                "품목코드",
                "품목명",
                "품종코드",
                "품종명",
                "등급코드(p_productrankcode)",
                "등급코드명",
            ]
        ]
