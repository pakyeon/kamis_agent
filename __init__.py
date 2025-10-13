# -*- coding: utf-8 -*-
"""
KAMIS 농축수산물 가격 정보 서비스

## 2개의 간단한 API

### 1. search() - 구조화된 데이터
다른 Agent가 데이터를 가공/분석할 때 사용

>>> from kamis_service import KamisService
>>> kamis = KamisService()
>>>
>>> # LLM이 품목 검색 + 가격 조회 → 구조화된 데이터
>>> data = kamis.search("사과 가격")
>>> print(data)
>>> {
>>>     "success": True,
>>>     "data": {
>>>         "query": "사과 가격",
>>>         "results": [...],
>>>         "tools_used": ["search_item", "daily_sales_list"]
>>>     }
>>> }

### 2. answer() - 자연어 답변
사용자에게 보여줄 답변이 필요할 때 사용

>>> answer = kamis.answer("오늘 사과 가격은?")
>>> print(answer)
>>> "2025-10-13 기준 사과(후지) 가격은 10kg당 15,000원입니다."

## 복잡한 질의도 동일!
Agent가 자동으로 여러 Tool을 조합

>>> # 데이터 분석용
>>> data = kamis.search("지난 3개월 서울/부산 배추 가격 비교")
>>>
>>> # 사용자 답변용
>>> answer = kamis.answer("지난 3개월 서울/부산 배추 가격 비교해줘")

## 다른 Agent에서 사용 예시

>>> class MealPlannerAgent:
...     def __init__(self):
...         self.kamis = KamisService()
...
...     def get_cheap_items(self):
...         # 구조화된 데이터로 받아서 처리
...         data = self.kamis.search("저렴한 채소 5개")
...         return data["data"]["results"]
...
...     def show_price_to_user(self, item):
...         # 사용자에게 보여줄 답변
...         return self.kamis.answer(f"{item} 가격 알려줘")
"""

from .service import KamisService
from .schemas import ItemInfo, PriceInfo, PriceTrend
from .exceptions import (
    KamisError,
    ConfigError,
    DatabaseError,
    APIError,
    ItemNotFoundError,
    ValidationError,
)

__version__ = "1.0.0"
__all__ = [
    "KamisService",
    "ItemInfo",
    "PriceInfo",
    "PriceTrend",
    "KamisError",
    "ConfigError",
    "DatabaseError",
    "APIError",
    "ItemNotFoundError",
    "ValidationError",
]
