# -*- coding: utf-8 -*-
"""SQL 쿼리 빌더"""

from typing import List, Tuple, Set


class QueryBuilder:
    """SQL 쿼리 생성"""

    LIVESTOCK_CATEGORY_CODE = "500"

    @staticmethod
    def build_product_search(
        keywords: List[str], limit: int = 50
    ) -> Tuple[str, List[str]]:
        """
        품목명으로 검색하는 쿼리

        Returns:
            (query, params)
        """
        conditions = " OR ".join(["product_name LIKE '%' || ? || '%'"] * len(keywords))
        query = f"""
            SELECT DISTINCT product_code
            FROM api_items
            WHERE {conditions}
            LIMIT ?
        """
        params = keywords + [str(limit)]
        return query, params

    @staticmethod
    def build_category_search(
        keywords: List[str], limit: int = 50
    ) -> Tuple[str, List[str]]:
        """부류명으로 검색하는 쿼리"""
        conditions = " OR ".join(["category_name LIKE '%' || ? || '%'"] * len(keywords))
        query = f"""
            SELECT DISTINCT category_code
            FROM api_items
            WHERE {conditions}
            LIMIT ?
        """
        params = keywords + [str(limit)]
        return query, params

    @staticmethod
    def build_filter_by_category(
        product_codes: Set[str], category_codes: Set[str]
    ) -> Tuple[str, List[str]]:
        """부류 코드로 품목 필터링"""
        prod_placeholders = ",".join("?" * len(product_codes))
        cat_placeholders = ",".join("?" * len(category_codes))

        query = f"""
            SELECT DISTINCT product_code
            FROM api_items
            WHERE product_code IN ({prod_placeholders})
              AND category_code IN ({cat_placeholders})
        """
        params = list(product_codes) + list(category_codes)
        return query, params

    @staticmethod
    def build_products_by_category(
        category_codes: Set[str], limit: int = 100
    ) -> Tuple[str, List[str]]:
        """부류 코드로 품목 조회"""
        placeholders = ",".join("?" * len(category_codes))
        query = f"""
            SELECT DISTINCT product_code
            FROM api_items
            WHERE category_code IN ({placeholders})
            LIMIT ?
        """
        params = list(category_codes) + [str(limit)]
        return query, params

    @staticmethod
    def build_hierarchy_search(
        product_code: str,
        kind_keywords: List[str],
        grade_keywords: List[str],
        is_livestock: bool,
    ) -> Tuple[str, List[str]]:
        """계층 정보 조회 (품종, 등급 필터 포함)"""
        if is_livestock:
            # 축산물
            query = """
                SELECT DISTINCT
                    category_code, category_name, product_code, product_name,
                    livestock_kind_code, kind_name,
                    p_periodProductList, p_periodProductName
                FROM api_items
                WHERE product_code = ? AND category_code = ?
            """
            params = [product_code, QueryBuilder.LIVESTOCK_CATEGORY_CODE]

            # 품종 필터
            if kind_keywords:
                kind_conditions = " OR ".join(
                    ["kind_name LIKE '%' || ? || '%'"] * len(kind_keywords)
                )
                query += f" AND ({kind_conditions})"
                params.extend(kind_keywords)

            # 등급 필터
            if grade_keywords:
                grade_conditions = []
                for _ in grade_keywords:
                    grade_conditions.extend(
                        [
                            "p_periodProductName LIKE '%' || ? || '%'",
                            "p_periodProductList LIKE '%' || ? || '%'",
                            "p_periodProductList = ?",
                        ]
                    )
                query += f" AND ({' OR '.join(grade_conditions)})"

                # 각 키워드에 대해 3개 조건씩
                for keyword in grade_keywords:
                    params.extend([keyword, keyword, keyword])
        else:
            # 일반 품목
            query = """
                SELECT DISTINCT
                    category_code, category_name, product_code, product_name,
                    kind_code, kind_name,
                    productrank_code, graderank_code, rank_name
                FROM api_items
                WHERE product_code = ? AND category_code != ?
            """
            params = [product_code, QueryBuilder.LIVESTOCK_CATEGORY_CODE]

            # 품종 필터
            if kind_keywords:
                kind_conditions = " OR ".join(
                    ["kind_name LIKE '%' || ? || '%'"] * len(kind_keywords)
                )
                query += f" AND ({kind_conditions})"
                params.extend(kind_keywords)

            # 등급 필터
            if grade_keywords:
                grade_conditions = []
                for _ in grade_keywords:
                    grade_conditions.extend(
                        [
                            "rank_name LIKE '%' || ? || '%'",
                            "graderank_code LIKE '%' || ? || '%'",
                        ]
                    )
                query += f" AND ({' OR '.join(grade_conditions)})"

                # 각 키워드에 대해 2개 조건씩
                for keyword in grade_keywords:
                    params.extend([keyword, keyword])

        return query, params
