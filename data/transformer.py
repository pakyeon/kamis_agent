# -*- coding: utf-8 -*-
"""데이터 병합 및 변환"""

import logging
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


class DataTransformer:
    """데이터 병합 및 변환"""

    # 최종 스키마 매핑
    SCHEMA_MAPPING = {
        "부류코드": "category_code",
        "부류명": "category_name",
        "품목코드": "product_code",
        "품목명": "product_name",
        "품종코드": "kind_code",
        "품종명": "kind_name",
        "축산_품종코드": "livestock_kind_code",
        "등급코드(p_productrankcode)": "productrank_code",
        "등급코드(p_graderank)": "graderank_code",
        "등급코드명": "rank_name",
        "등급코드(periodProductList)": "p_periodProductList",
        "등급코드(periodProductName)": "p_periodProductName",
    }

    def transform(self, sheets_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        시트 데이터 병합 및 변환

        Args:
            sheets_data: 시트별 데이터프레임

        Returns:
            최종 데이터프레임
        """
        # 1. 데이터 타입 통일
        for name, df in sheets_data.items():
            sheets_data[name] = self._normalize_types(df)
            logger.info(f"{name}: {len(df)}행")

        # 2. 병합
        merged = self._merge_all(sheets_data)
        logger.info(f"병합 완료: {len(merged)}행")

        # 3. 스키마 매핑
        final = self._map_schema(merged)
        logger.info(f"스키마 매핑 완료: {len(final)}행")

        # 4. 중복 제거
        final = final.drop_duplicates().reset_index(drop=True)
        logger.info(f"중복 제거 후: {len(final)}행")

        return final

    def _normalize_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 타입 통일 (조인 키는 string)"""
        key_columns = ["부류코드", "품목코드", "품종코드", "축산_품종코드"]

        for col in key_columns:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace("nan", pd.NA)

        return df

    def _merge_all(self, sheets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """모든 시트 병합"""
        # 산물 데이터 (기본)
        general = sheets["산물"].copy()

        # 부류명 lookup
        general = self._lookup_merge(
            general, sheets["부류"], ["부류코드"], {"부류명": "부류명"}
        )

        # 품목명 lookup
        general = self._lookup_merge(
            general,
            sheets["품목"][["품목코드", "품목명"]].drop_duplicates(),
            ["품목코드"],
            {"품목명": "품목명"},
        )

        # 품종명 lookup
        general = self._lookup_merge(
            general,
            sheets["품종"][["품목코드", "품종코드", "품종명"]].drop_duplicates(),
            ["품목코드", "품종코드"],
            {"품종명": "품종명"},
        )

        # 등급 정보 lookup
        if not sheets["등급"].empty:
            general = self._lookup_merge(
                general,
                sheets["등급"],
                ["등급코드(p_productrankcode)"],
                {
                    "등급코드(p_graderank)": "등급코드(p_graderank)",
                    "등급코드명": "등급코드명",
                },
            )

        # 축산물 컬럼 추가 (빈 값)
        general["축산_품종코드"] = pd.NA
        general["등급코드(periodProductList)"] = pd.NA
        general["등급코드(periodProductName)"] = pd.NA

        # 축산물 데이터 (품목명 lookup)
        livestock = sheets["축산물"].copy()
        livestock = self._lookup_merge(
            livestock,
            sheets["품목"][["품목코드", "품목명"]].drop_duplicates(),
            ["품목코드"],
            {"품목명": "품목명"},
        )

        # 일반 등급 컬럼 추가 (빈 값)
        livestock["품종코드"] = pd.NA
        livestock["등급코드(p_productrankcode)"] = pd.NA
        livestock["등급코드(p_graderank)"] = pd.NA
        livestock["등급코드명"] = pd.NA

        # 기본 품목 데이터 (산물/축산물에 없는 것들)
        base = self._create_base_items(sheets, general, livestock)

        # 통합
        final_columns = [
            "부류코드",
            "부류명",
            "품목코드",
            "품목명",
            "품종코드",
            "품종명",
            "축산_품종코드",
            "등급코드(p_productrankcode)",
            "등급코드(p_graderank)",
            "등급코드명",
            "등급코드(periodProductList)",
            "등급코드(periodProductName)",
        ]

        # 각 데이터프레임에 모든 컬럼 추가
        for df in [general, livestock, base]:
            for col in final_columns:
                if col not in df.columns:
                    df[col] = pd.NA

        return pd.concat(
            [general[final_columns], livestock[final_columns], base[final_columns]],
            ignore_index=True,
        )

    def _lookup_merge(
        self,
        data_df: pd.DataFrame,
        lookup_df: pd.DataFrame,
        merge_keys: list,
        lookup_cols: dict,
    ) -> pd.DataFrame:
        """Lookup 병합 (데이터 손실 없이)"""
        result = data_df.merge(
            lookup_df[merge_keys + list(lookup_cols.keys())],
            on=merge_keys,
            how="left",
            suffixes=("", "_lookup"),
        )

        # Lookup 값으로 빈 값 채우기
        for src_col, dst_col in lookup_cols.items():
            lookup_col = (
                f"{src_col}_lookup"
                if f"{src_col}_lookup" in result.columns
                else src_col
            )
            if dst_col not in result.columns:
                result[dst_col] = result[lookup_col]
            else:
                result[dst_col] = result[dst_col].fillna(result[lookup_col])

        # Lookup 접미사 컬럼 제거
        cols_to_drop = [c for c in result.columns if c.endswith("_lookup")]
        return result.drop(columns=cols_to_drop)

    def _create_base_items(
        self,
        sheets: Dict[str, pd.DataFrame],
        general: pd.DataFrame,
        livestock: pd.DataFrame,
    ) -> pd.DataFrame:
        """기본 품목/품종 데이터 생성"""
        # 품목 기반 데이터
        base = sheets["품목"].copy()

        # 부류명 lookup
        base = self._lookup_merge(
            base, sheets["부류"], ["부류코드"], {"부류명": "부류명"}
        )

        # 품종이 있는 품목은 품종별로 확장
        items_with_kind = base.merge(
            sheets["품종"][["품목코드", "품종코드", "품종명"]],
            on="품목코드",
            how="inner",
        )

        # 품종이 없는 품목
        items_without_kind = base[
            ~base["품목코드"].isin(items_with_kind["품목코드"])
        ].copy()
        items_without_kind["품종코드"] = pd.NA
        items_without_kind["품종명"] = pd.NA

        # 합치기
        base = pd.concat([items_with_kind, items_without_kind], ignore_index=True)

        # 이미 산물/축산물에 있는 조합 제외
        sanmul_combos = set(
            general[["부류코드", "품목코드", "품종코드"]]
            .dropna(subset=["품목코드"])
            .apply(lambda x: f"{x['부류코드']}|{x['품목코드']}|{x['품종코드']}", axis=1)
        )
        livestock_combos = set(
            livestock[["부류코드", "품목코드"]].apply(
                lambda x: f"{x['부류코드']}|{x['품목코드']}|", axis=1
            )
        )

        base["_combo"] = base.apply(
            lambda x: f"{x['부류코드']}|{x['품목코드']}|{x['품종코드']}", axis=1
        )
        base = base[
            ~base["_combo"].isin(sanmul_combos) & ~base["_combo"].isin(livestock_combos)
        ].drop("_combo", axis=1)

        return base

    def _map_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """최종 스키마로 매핑"""
        result = pd.DataFrame()

        for src_col, dst_col in self.SCHEMA_MAPPING.items():
            if src_col in df.columns:
                result[dst_col] = df[src_col].astype("string")
            else:
                result[dst_col] = pd.Series(dtype="string")

        return result
