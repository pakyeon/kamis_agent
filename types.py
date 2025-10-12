# -*- coding: utf-8 -*-
"""KAMIS Service 타입 정의"""

from typing import TypedDict, Literal, Optional, List, Dict


# 품목 정보
class ItemInfo(TypedDict, total=False):
    """품목 기본 정보"""

    product_code: str
    product_name: str
    category: str
    category_code: str
    kind: Optional[str]
    kind_code: Optional[str]
    grade: Optional[str]
    grade_code: Optional[str]


# 가격 정보 (KAMIS API는 모든 값을 string으로 반환)
class PriceInfo(TypedDict, total=False):
    """가격 정보 (모든 필드는 string)"""

    item: str
    product_code: str
    price: str  # KAMIS API는 string으로 반환
    unit: str
    date: str
    region: str
    region_code: str
    change_rate: Optional[str]  # 전일 대비 변화율 (예: "5.2")
    change_amount: Optional[str]  # 전일 대비 변화량


# 가격 추이
class PriceTrendData(TypedDict):
    """가격 추이 데이터"""

    date: str
    price: str  # string


class PriceStatistics(TypedDict):
    """가격 통계"""

    avg: str
    min: str
    max: str
    trend: Literal["increasing", "decreasing", "stable"]


class PriceTrend(TypedDict):
    """가격 추이 전체"""

    item: str
    product_code: str
    period: Dict[str, str]  # {"start": "...", "end": "..."}
    prices: List[PriceTrendData]
    statistics: PriceStatistics


# 상수
TrendDirection = Literal["increasing", "decreasing", "stable"]
Category = Literal["식량작물", "채소류", "과일류", "축산물", "수산물", "특용작물"]
ProductClass = Literal["01", "02"]  # 01=소매, 02=도매
