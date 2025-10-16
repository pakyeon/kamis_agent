# KAMIS Agent

> 🌾 KAMIS 농축수산물 가격 정보를 LLM Agent로 제공하는 Python 서비스

KAMIS (한국농수산식품유통공사) Open API와 LangGraph를 결합하여, 자연어로 농축수산물 가격 정보를 조회하고 분석할 수 있는 지능형 에이전트입니다.

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## ✨ 주요 기능

- 🤖 **자연어 질의**: 복잡한 API 파라미터 없이 자연어로 질문
- 🔍 **지능형 검색**: LLM이 품목명을 자동으로 매칭
- 📊 **구조화된 데이터**: 다른 Agent가 사용하기 쉬운 데이터 제공
- 💬 **자연어 답변**: 사용자 친화적인 답변 생성
- 🔧 **17개 KAMIS API 통합**: 11개 활성화, 6개 비활성화 (API 오류)

## 🚀 빠른 시작

### 설치

```bash
pip install -r requirements.txt
```

### 환경 설정

`.env` 파일 생성:

```env
# KAMIS API 인증 (필수)
KAMIS_CERT_KEY=your_cert_key
KAMIS_CERT_ID=your_cert_id

# OpenAI API (필수)
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-5-mini

# 선택사항
DB_PATH=kamis_api_list.db
DB_MAX_AGE_HOURS=24
REASONING_EFFORT=minimal
```

### 기본 사용법

```python
from kamis_agent import KamisService

# 서비스 초기화
kamis = KamisService()

# 1️⃣ 구조화된 데이터 조회 (다른 Agent용)
data = kamis.search("오늘 사과 가격")
print(data)
# {
#   "success": true,
#   "query": "사과 가격",
#   "items": [
#     {
#       "category": {
#         "code": "400",
#         "name": "과일류"
#       },
#     }
#   ]
# }

# 2️⃣ 자연어 답변 (사용자용)
answer = kamis.answer("오늘 사과 가격은?")
print(answer)
# "2025-10-13 기준 사과(후지) 가격은 10kg당 15,000원입니다."
```

## 📚 API 문서

### `search(natural_query: str) -> Dict`

자연어 쿼리로 검색하고 구조화된 데이터를 반환합니다.

**용도**: 다른 Agent가 데이터를 가공/분석할 때

```python
# 간단한 질의
data = kamis.search("사과 가격")

# 복잡한 질의 (Agent가 자동으로 여러 Tool 사용)
data = kamis.search("지난 3개월 서울과 부산의 배추 가격 비교")

# 반환 형식
{
  "success": true,
  "query": "사과 가격",
  "items": [
    {
      "category": {
        "code": "400",
        "name": "과일류"
      },
      "product": {
        "code": "411",
        "name": "사과"
      },
      "kind": {
        "code": "6",
        "name": "쓰가루아오리"
      },
      "grade": {
        "productrank_code": "4",
        "graderank_code": "1",
        "name": "상품"
      },
      "market": {
        "code": "01",
        "name": "소매"
      }
    }, ...
  ]
}
```

### `answer(natural_query: str) -> str`

자연어 쿼리로 검색하고 자연어 답변을 생성합니다.

**용도**: 사용자에게 직접 보여줄 답변이 필요할 때

```python
# 간단한 질의
answer = kamis.answer("오늘 1++등급 돼지고기 가격은?")

# 복잡한 질의
answer = kamis.answer("최근 한 달간 배추 가격 추이를 설명해줘")

# 반환 형식
"2025-10-13 기준 돼지고기(삼겹살) 가격은 100g당 2,500원입니다."
```

## 🎯 다른 Agent에서 사용 예시

### 식단 계획 Agent

```python
from kamis_agent import KamisService

class MealPlannerAgent:
    """예산 기반 식단 계획 Agent"""
    
    def __init__(self):
        self.kamis = KamisService()
    
    def get_cheap_items(self, budget: int):
        """저렴한 품목 찾기"""
        # 구조화된 데이터로 받아서 처리
        data = self.kamis.search("저렴한 채소 5개")
        
        if data.get("success"):
            return self._filter_by_budget(data["data"]["results"], budget)
        return []
    
    def show_price_to_user(self, item: str):
        """사용자에게 가격 정보 제공"""
        # 자연어 답변 생성
        return self.kamis.answer(f"{item} 가격 알려줘")

# 사용
planner = MealPlannerAgent()
cheap_items = planner.get_cheap_items(budget=50000)
message = planner.show_price_to_user("배추")
```

### 재고 관리 Agent

```python
class InventoryAgent:
    """재고 관리 및 구매 최적화 Agent"""
    
    def __init__(self):
        self.kamis = KamisService()
    
    def check_price_trend(self, item: str, days: int = 30):
        """가격 추이 분석"""
        data = self.kamis.search(f"최근 {days}일 {item} 가격 추이")
        return self._analyze_trend(data)
    
    def compare_regions(self, item: str, regions: list):
        """지역별 가격 비교"""
        query = f"{', '.join(regions)} {item} 가격 비교"
        data = self.kamis.search(query)
        return self._find_cheapest_region(data)
```

## 🔌 지원 KAMIS API 목록 (17개)

| 번호 | API 명 | 엔드포인트 | 활성화 | 설명 |
|------|--------|-----------|:------:|------|
| 1 | 일별 부류별 도.소매가격정보 | `daily_by_category` | ✅ | 특정일의 부류별 전체 품목 가격 조회 |
| 2 | 일별 품목별 도·소매가격정보 | `daily_by_item_period` | ❌ | 일별 품목별 가격 (기간, 최대 1년) |
| 3 | 월별 도.소매가격정보 | `monthly_sales` | ✅ | 특정 품목의 월별 가격 (최대 3년) |
| 4 | 연도별 도.소매가격정보 | `yearly_sales` | ✅ | 특정 품목의 연도별 가격 |
| 5 | 친환경농산물 가격정보('05~'20.3.) | `old_eco_period` | ❌ | 친환경 품목 기간별 가격 (2005-2020.3) |
| 6 | 최근일자 도.소매가격정보 | `daily_sales_list` | ✅ | 최근 거래일 전체 품목 가격 |
| 7 | 최근 가격추이 조회 | `recent_price_trend` | ✅ | 단기 가격 추이 (작년/평년 비교) |
| 8 | 월평균 가격추이 조회 | `monthly_price_trend` | ❌ | 월별 가격 추이 분석 |
| 9 | 연평균 가격추이 조회 | `yearly_price_trend` | ❌ | 연별 가격 추이 분석 |
| 10 | 최근일자 지역별 도.소매가격정보 | `daily_county` | ✅ | 특정 지역의 전체 품목 가격 |
| 11 | 친환경농산물 품목별 가격정보('05~'20.3.) | `old_eco_item` | ❌ | 친환경 품목 특정일 가격 (2005-2020.3) |
| 12 | 친환경농산물 가격정보('20.4~) | `new_eco_period` | ✅ | 친환경 품목 기간별 가격 (2020.4 이후) |
| 13 | 친환경농산물 품목별가격정보('20.4~) | `new_eco_item` | ✅ | 친환경 품목 특정일 가격 (2020.4 이후) |
| 14 | 지역별 품목별 도.소매가격정보 | `region_item` | ✅ | 특정 품목의 지역별 가격 비교 |
| 15 | 농축수산물 품목 및 등급 코드표 | `product_info` | ✅ | 전체 품목 코드 매핑 정보 |
| 16 | 일별 품목별 도매 가격자료 | `period_wholesale` | ✅ | 도매시장 일별 가격 (최대 1년) |
| 17 | 일별 품목별 소매 가격자료 | `period_retail` | ✅ | 소매시장 일별 가격 (최대 1년) |

**활성화 현황**: 11개 활성화 / 6개 비활성화

## 🏗️ 프로젝트 구조

```
kamis_agent/
├── __init__.py              # Public API (search, answer)
├── service.py               # KamisService 메인 클래스
├── types.py                 # 타입 정의 (ItemInfo, PriceInfo 등)
├── exceptions.py            # 예외 클래스
├── config.py                # 환경 설정 관리
│
├── core/
│   ├── __init__.py
│   └── client.py           # KAMIS API HTTP 클라이언트
│
├── search/
│   ├── __init__.py
│   ├── searcher.py         # 계층적 검색 엔진 (LLM 통합)
│   ├── text_processor.py   # 한국어 형태소 분석
│   ├── db_manager.py       # SQLite 연결 관리
│   └── query_builder.py    # SQL 쿼리 생성
│
├── data/
│   ├── __init__.py
│   ├── manager.py          # 데이터 업데이트 관리
│   ├── downloader.py       # KAMIS 문서(Excel) 다운로드
│   ├── extractor.py        # 6개 시트 데이터 추출
│   ├── transformer.py      # 데이터 병합 및 변환
│   └── loader.py           # SQLite DB 적재
│
└── agent/
    ├── __init__.py
    ├── executor.py         # LangGraph Agent 실행기
    ├── tool_factory.py     # LangChain Tool 생성
    ├── prompts.py          # 시스템 프롬프트
    └── api_endpoints.py    # 17개 KAMIS API 정의
```

## 🔑 주요 특징

### 1. 자동 품목 매칭

```python
# "사과" 입력 → 자동으로 품목코드 매칭
data = kamis.search("후지 사과 상품")
# LLM이 자동으로 품목코드, 품종, 등급 추출
```

### 2. 복잡한 질의 자동 처리

```python
# Agent가 자동으로 여러 API를 조합
answer = kamis.answer("지난 3개월 서울, 부산, 대구의 배추 가격을 비교하고 가장 저렴한 지역을 알려줘")
# → search_item + region_item + 분석
```

### 3. 자동 DB 업데이트

```python
# TTL(기본 24시간) 기반 자동 갱신
kamis = KamisService()  # DB가 오래되면 자동 업데이트
```

### 4. API 오류 자동 대응

```python
# 비활성화된 API를 자동으로 대체 API로 처리
data = kamis.search("지난 한 달 배추 가격 추이")
# daily_by_item_period(비활성) → period_retail(활성) 자동 전환
```

## 🛠️ 기술 스택

- **Python 3.10+**
- **LangChain**
- **LangGraph**
- **OpenAI API LLM**
- **SQLite**
- **Kiwipiepy**: 한국어 형태소 분석
- **Pandas**

## 🔒 환경 변수

| 변수 | 필수 | 기본값 | 설명 |
|-----|------|--------|------|
| `KAMIS_CERT_KEY` | ✅ | - | KAMIS API 인증키 |
| `KAMIS_CERT_ID` | ✅ | - | KAMIS API 인증ID |
| `OPENAI_API_KEY` | ✅ | - | OpenAI API 키 |
| `OPENAI_MODEL` | ❌ | gpt-5-mini | 사용할 모델 |
| `REASONING_EFFORT` | ❌ | minimal | 추론 정도 (minimal/low/medium/high) |
| `DB_PATH` | ❌ | kamis_api_list.db | DB 파일 경로 |
| `DB_MAX_AGE_HOURS` | ❌ | 24 | DB 최대 유효 시간 |

## 📝 KAMIS API 인증키 발급

1. [KAMIS 오픈 API](https://www.kamis.or.kr/customer/reference/openapi_list.do) 접속
2. 회원가입 및 로그인
3. API 신청
4. 발급된 인증키를 `.env`에 설정

## ⚠️ 알려진 이슈

### KAMIS API 오류로 인한 비활성화 (6개)

일부 KAMIS API에서 파라미터가 정상적으로 작동하지 않는 문제로 인해 비활성화되었습니다:

<details>
<summary><b>#2 일별 품목별 도·소매가격정보</b> (<code>daily_by_item_period</code>)</summary>

- **문제점**: `p_startday`, `p_endday` 파라미터가 반영되지 않음
- **대체 방법**: `period_wholesale` (도매) + `period_retail` (소매) 사용
- **영향**: 도매/소매 구분 없이 기간별 조회 불가

</details>

<details>
<summary><b>#5 친환경농산물 가격정보('05~'20.3.)</b> (<code>old_eco_period</code>)</summary>

- **문제점**: `p_startday`, `p_endday` 파라미터가 반영되지 않음
- **대체 방법**: `new_eco_period` (2020.4 이후 데이터만 제공)
- **영향**: 2005-2020.3 기간 친환경 농산물 기간별 조회 불가

</details>

<details>
<summary><b>#8 월평균 가격추이 조회</b> (<code>monthly_price_trend</code>)</summary>

- **문제점**: `p_regday` 파라미터가 반영되지 않고 항상 최근 데이터만 반환
- **대체 방법**: `monthly_sales` 데이터를 LLM이 분석하여 추이 제공
- **영향**: 특정 시점 기준 월평균 추이 직접 조회 불가

</details>

<details>
<summary><b>#9 연평균 가격추이 조회</b> (<code>yearly_price_trend</code>)</summary>

- **문제점**: `p_regday` 파라미터가 반영되지 않고 항상 최근 데이터만 반환
- **대체 방법**: `yearly_sales` 데이터를 LLM이 분석하여 추이 제공
- **영향**: 특정 시점 기준 연평균 추이 직접 조회 불가

</details>

<details>
<summary><b>#11 친환경농산물 품목별 가격정보('05~'20.3.)</b> (<code>old_eco_item</code>)</summary>

- **문제점**: 대부분의 날짜에 데이터가 없으며 특정 일자만 존재
- **대체 방법**: `new_eco_item` (2020.4 이후 데이터만 제공)
- **영향**: 2005-2020.3 기간 친환경 농산물 특정일 조회 불가

</details>

### 참고사항
- 비활성화된 API들은 `agent/api_endpoints.py`에 주석으로 보존되어 있습니다
- KAMIS API가 수정되면 언제든지 재활성화 가능합니다
- 대체 API들을 통해 동일한 기능을 제공하고 있습니다
- 이러한 문제들은 KAMIS API 측의 이슈이며, 본 프로젝트의 문제가 아닙니다

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.