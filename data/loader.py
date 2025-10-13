# -*- coding: utf-8 -*-
"""SQLite 데이터 로더"""

import logging
import sqlite3
from typing import Dict
import pandas as pd

from ..exceptions import DatabaseError

logger = logging.getLogger(__name__)


class DataLoader:
    """SQLite DB 적재"""

    # 스키마 정의
    SCHEMA = {
        "category_code": "TEXT NOT NULL",
        "category_name": "TEXT NOT NULL",
        "product_code": "TEXT NOT NULL",
        "product_name": "TEXT NOT NULL",
        "kind_code": "TEXT",
        "kind_name": "TEXT",
        "livestock_kind_code": "TEXT",
        "productrank_code": "TEXT",
        "graderank_code": "TEXT",
        "rank_name": "TEXT",
        "p_periodProductList": "TEXT",
        "p_periodProductName": "TEXT",
    }

    def load(self, df: pd.DataFrame, db_path: str, table_name: str) -> None:
        """
        데이터를 SQLite DB에 적재

        Args:
            df: 데이터프레임
            db_path: DB 파일 경로
            table_name: 테이블명
        """
        # 필수 컬럼 검증
        required = ["category_code", "category_name", "product_code", "product_name"]
        self._validate_required_columns(df, required)

        try:
            with sqlite3.connect(db_path) as conn:
                # 기존 테이블 삭제
                conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')

                # 테이블 생성
                self._create_table(conn, table_name)

                # 데이터 삽입
                df.to_sql(table_name, conn, if_exists="append", index=False)

                # 인덱스 생성
                self._create_indexes(conn, table_name)

                logger.info(f"DB 적재 완료: {len(df)}행 → {table_name}")

        except sqlite3.Error as e:
            raise DatabaseError(f"DB 적재 실패: {e}")

    def _validate_required_columns(self, df: pd.DataFrame, required: list) -> None:
        """필수 컬럼 검증"""
        # 빈 값 체크
        for col in required:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip()

        df[required] = df[required].replace("", pd.NA)

        before = len(df)
        df.dropna(subset=required, inplace=True)
        dropped = before - len(df)

        if dropped > 0:
            logger.warning(f"필수 컬럼 누락으로 {dropped}행 제거")

    def _create_table(self, conn: sqlite3.Connection, table_name: str) -> None:
        """테이블 생성"""
        columns = ", ".join([f'"{k}" {v}' for k, v in self.SCHEMA.items()])
        conn.execute(f'CREATE TABLE "{table_name}" ({columns})')

    def _create_indexes(self, conn: sqlite3.Connection, table_name: str) -> None:
        """인덱스 생성"""
        indexes = [
            ("idx_category_code", "category_code"),
            ("idx_product_code", "product_code"),
            ("idx_kind_code", "kind_code"),
            ("idx_livestock_kind_code", "livestock_kind_code"),
            ("idx_productrank_code", "productrank_code"),
            ("idx_graderank_code", "graderank_code"),
        ]

        for idx_name, col_name in indexes:
            conn.execute(
                f'CREATE INDEX IF NOT EXISTS {idx_name} ON "{table_name}"("{col_name}")'
            )
