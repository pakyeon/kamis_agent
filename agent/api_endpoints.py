# -*- coding: utf-8 -*-
"""
KAMIS API 엔드포인트 정의

=== 이 파일의 역할 ===
LangChain Tool의 description으로 사용되는 메타데이터 정의

- LLM이 Tool 선택 시 직접 참고하는 정보 제공
- tool_factory.py에서 자동으로 Tool 생성 시 사용
- 각 API의 목적, 사용 시점, 파라미터 설명 포함

=== 정보 전달 흐름 ===
1. 이 파일의 desc → StructuredTool의 description
2. LLM이 Tool 목록을 볼 때 이 description 참조
3. prompts.py의 전략 가이드와 함께 조합되어 의사결정

=== 관련 파일 ===
- tool_factory.py: 이 데이터를 읽어 LangChain Tool 생성
- prompts.py: 고수준 전략 및 제약사항 제공
- searcher.py: resolve_query Tool의 실제 구현

=== 유지보수 가이드 ===
- API 추가/변경 시 이 파일만 수정하면 됨
- desc는 간결하되 명확하게 작성 (LLM이 선택 기준으로 사용)
- 사용 시점, 제약사항을 명확히 표현
"""

from typing import Dict, Any

# 17개 KAMIS API 정의
API_ENDPOINTS: Dict[str, Dict[str, Any]] = {
    "daily_by_category": {
        "action": "dailyPriceByCategoryList",
        "desc": "특정일의 부류별 전체 품목 가격정보 조회. 채소류/과일류 등 카테고리 단위 일괄 조회. 부류별 전체 품목 파악시에만 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
            "p_product_cls_code": "구분(01=소매, 02=도매)",
            "p_item_category_code": {"desc": "부류코드(100~600)", "required": True},
            "p_country_code": "시군구코드",
            "p_regday": {"desc": "검색일자 YYYY-MM-DD", "required": True},
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    # ========================================
    # ❌ KAMIS API 오류로 비활성화
    # 문제: p_startday, p_endday 파라미터가 반영되지 않음
    # 대체: period_wholesale (도매), period_retail (소매) 사용
    # ========================================
    # "daily_by_item_period": {
    #     "action": "periodProductList",
    #     "desc": "일별 품목별 가격(기간, 당일로부터 1년간의 데이터만 제공). 특정 품목 일자별 추이",
    #     "fields": {
    #         "p_returntype": "반환 형식(json, xml)",
    #         "p_startday": "시작일 YYYY-MM-DD",
    #         "p_endday": "종료일 YYYY-MM-DD",
    #         "p_productclscode": "구분(01=소매, 02=도매)",
    #         "p_itemcategorycode": "부류코드",
    #         "p_itemcode": {"desc": "품목코드", "required": True},
    #         "p_kindcode": "품종코드",
    #         "p_productrankcode": "등급코드",
    #         "p_countrycode": "시군구코드",
    #         "p_convert_kg_yn": "kg환산(Y/N)",
    #     },
    # },
    "monthly_sales": {
        "action": "monthlySalesList",
        "desc": "특정 품목의 월별 가격정보 조회. 최대 3년 이내 월단위 집계 데이터 제공. 월별 정보 분석시 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
            "p_yyyy": "기준연도",
            "p_period": "조회기간(년수, default: 3)",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": {"desc": "품목코드", "required": True},
            "p_kindcode": "품종코드",
            "p_graderank": "등급순위",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "yearly_sales": {
        "action": "yearlySalesList",
        "desc": "특정 품목의 연도별 가격정보 조회. 연단위 집계 데이터 제공. 1년 초과 장기 정보 분석시 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
            "p_yyyy": "기준연도",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": {"desc": "품목코드", "required": True},
            "p_kindcode": "품종코드",
            "p_graderank": "등급순위",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    # ========================================
    # ❌ KAMIS API 오류로 비활성화 (2005~2020.3 데이터)
    # 문제: p_startday, p_endday 파라미터가 반영되지 않음
    # 대체: new_eco_period (2020.4 이후 데이터만 제공)
    # ========================================
    # "old_eco_period": {
    #     "action": "periodNaturePriceList",
    #     "desc": "친환경 농산물 가격정보(2005~2020.3)",
    #     "fields": {
    #         "p_returntype": "반환 형식(json, xml)",
    #         "p_startday": {"desc": "시작일", "required": True},
    #         "p_endday": {"desc": "종료일", "required": True},
    #         "p_itemcategorycode": "부류코드",
    #         "p_itemcode": {"desc": "품목코드", "required": True},
    #         "p_kindcode": "품종코드",
    #         "p_productrankcode": "등급코드",
    #         "p_countrycode": "시군구코드",
    #         "p_convert_kg_yn": "kg환산(Y/N)",
    #     },
    # },
    "daily_sales_list": {
        "action": "dailySalesList",
        "desc": "최근 거래일의 전체 품목 가격정보 조회. 파라미터 없이 모든 품목 일괄 반환. 전체 품목 파악시에만 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
        },
    },
    "recent_price_trend": {
        "action": "recentlyPriceTrendList",
        "desc": "특정 품목의 단기 가격 동향 정보. 작년 동기 및 평년 동기 비교 데이터 제공. 단기 동향 파악시 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
            "p_regday": "검색일자 YYYY-MM-DD",
            "p_productno": {"desc": "품목코드", "required": True},
        },
    },
    # ========================================
    # ❌ KAMIS API 오류로 비활성화
    # 문제: p_regday 파라미터가 반영되지 않음
    # 대체: monthly_sales 사용 (raw 데이터를 LLM이 분석하여 추이 제공)
    # ========================================
    # "monthly_price_trend": {
    #     "action": "monthlyPriceTrendList",
    #     "desc": "월평균 동향. 월별 트렌드 분석",
    #     "fields": {
    #         "p_returntype": "반환 형식(json, xml)",
    #         "p_productno": {"desc": "품목코드", "required": True},
    #         "p_regday": "검색일자",
    #     },
    # },
    # ========================================
    # ❌ KAMIS API 오류로 비활성화
    # 문제: p_regday 파라미터가 반영되지 않음
    # 대체: yearly_sales 사용 (raw 데이터를 LLM이 분석하여 추이 제공)
    # ========================================
    # "yearly_price_trend": {
    #     "action": "yearlyPriceTrendList",
    #     "desc": "연평균 동향. 장기 트렌드 분석",
    #     "fields": {
    #         "p_returntype": "반환 형식(json, xml)",
    #         "p_productno": {"desc": "품목코드", "required": True},
    #         "p_regday": "검색일자",
    #     },
    # },
    "daily_county": {
        "action": "dailyCountyList",
        "desc": "특정 지역의 최근 거래일 전체 품목 가격정보 조회. 해당 지역 축산물을 제외한 모든 품목 일괄 반환. 지역별 전체 현황 파악시 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
            "p_countrycode": {"desc": "시군구코드", "required": True},
        },
    },
    # ========================================
    # ❌ KAMIS API 오류로 비활성화 (2005~2020.3 데이터)
    # 문제: 대부분의 날짜에 데이터가 없음 (특정일자만 존재)
    # 대체: new_eco_item (2020.4 이후 데이터만 제공)
    # ========================================
    # "old_eco_item": {
    #     "action": "NaturePriceList",
    #     "desc": "친환경가격(2005-2020.3). 특정일자",
    #     "fields": {
    #         "p_returntype": "반환 형식(json, xml)",
    #         "p_regday": "검색일자",
    #         "p_itemcategorycode": "부류코드",
    #         "p_itemcode": {"desc": "품목코드", "required": True},
    #         "p_kindcode": "품종코드",
    #         "p_productrankcode": "등급코드",
    #         "p_countrycode": "시군구코드",
    #         "p_convert_kg_yn": "kg환산(Y/N)",
    #     },
    # },
    "new_eco_period": {
        "action": "periodEcoPriceList",
        "desc": "친환경농산물 특정 품목 기간별 가격정보(2020.4~현재) 조회. 특정 기간 일별 데이터 제공. 친환경 품목 기간 조회시 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
            "p_startday": "시작일",
            "p_endday": "종료일",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": {"desc": "품목코드", "required": True},
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "new_eco_item": {
        "action": "EcoPriceList",
        "desc": "친환경농산물 특정 품목 특정일 가격정보(2020.4~현재) 조회. 단일일자 데이터 제공. 친환경 품목 특정일 조회시 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
            "p_regday": "검색일자",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": {"desc": "품목코드", "required": True},
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_countrycode": "시군구코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "region_item": {
        "action": "ItemInfo",
        "desc": "특정 품목의 지역별 가격정보 조회. 시군구코드 미입력시 전국 전체 지역 반환. 지역 비교시 각 지역별 개별 호출 필요",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
            "p_productclscode": "구분(01=소매,02=도매)",
            "p_regday": "검색일자",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": {"desc": "품목코드", "required": True},
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_countrycode": "시군구코드(필수)",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "product_info": {
        "action": "productInfo",
        "desc": "전체 품목 코드표 정보. 품목/품종/등급 코드 매핑 제공 (축산물 제외). resolve_query 실패시에만 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
        },
    },
    "period_wholesale": {
        "action": "periodWholesaleProductList",
        "desc": "도매시장 특정 품목 일별 가격정보 조회. 도매가 전용 일별 데이터 제공. 도매 시장 특정 품목 일별 조회시 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
            "p_startday": "시작일",
            "p_endday": "종료일",
            "p_countrycode": "시군구코드",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": {"desc": "품목코드", "required": True},
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
    "period_retail": {
        "action": "periodRetailProductList",
        "desc": "소매시장 특정 품목 일별 가격정보 조회. 소매가 전용 일별 데이터 제공. 소매 시장 특정 품목 일별 조회시 사용",
        "fields": {
            "p_returntype": "반환 형식(json, xml)",
            "p_startday": "시작일",
            "p_endday": "종료일",
            "p_countrycode": "시군구코드",
            "p_itemcategorycode": "부류코드",
            "p_itemcode": {"desc": "품목코드", "required": True},
            "p_kindcode": "품종코드",
            "p_productrankcode": "등급코드",
            "p_convert_kg_yn": "kg환산(Y/N)",
        },
    },
}
