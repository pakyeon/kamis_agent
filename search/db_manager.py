# -*- coding: utf-8 -*-
"""데이터베이스 연결 관리"""

import sqlite3
import atexit
from typing import List, Dict, Any, Optional
from contextlib import contextmanager

from ..exceptions import DatabaseError


class DatabaseManager:
    """SQLite 데이터베이스 연결 관리"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._connection: Optional[sqlite3.Connection] = None

        # 초기화
        self._connect()
        self._ensure_indexes()

        # 프로그램 종료 시 자동 정리
        atexit.register(self.close)

    def _connect(self) -> None:
        """데이터베이스 연결"""
        try:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row
        except sqlite3.Error as e:
            raise DatabaseError(f"데이터베이스 연결 실패: {e}")

    def _ensure_indexes(self) -> None:
        """필요한 인덱스 생성"""
        if not self._connection:
            return

        cursor = self._connection.cursor()
        indexes = [
            ("idx_category_code", "category_code"),
            ("idx_product_code", "product_code"),
            ("idx_kind_code", "kind_code"),
            ("idx_livestock_kind", "livestock_kind_code"),
            ("idx_productrank", "productrank_code"),
            ("idx_graderank", "graderank_code"),
        ]

        for idx_name, col_name in indexes:
            cursor.execute(
                f"CREATE INDEX IF NOT EXISTS {idx_name} ON api_items({col_name})"
            )

        self._connection.commit()

    @contextmanager
    def cursor(self):
        """커서 컨텍스트 매니저"""
        if not self._connection:
            raise DatabaseError("데이터베이스가 연결되지 않았습니다.")

        cursor = self._connection.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

    def execute(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        """쿼리 실행"""
        with self.cursor() as cur:
            try:
                cur.execute(query, params)
                return cur.fetchall()
            except sqlite3.Error as e:
                raise DatabaseError(f"쿼리 실행 실패: {e}")

    def close(self) -> None:
        """연결 종료"""
        if self._connection:
            self._connection.close()
            self._connection = None
