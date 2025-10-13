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
- 🔧 **17개 KAMIS API 통합**: 가격, 추이, 지역 비교 등

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
#     "success": True,
#     "data": {
#         "query": "오늘 사과 가격",
#         "results": [...]
#     }
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
    "success": True,
    "data": {
        "query": "...",
        "results": [...],
        "tools_used": ["search_item", "daily_by_item_period"]
    },
    "message_count": 5
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

## 🏗️ 프로젝트 구조

```
kamis_agent/
├── __init__.py              # Public API (search, answer)
├── service.py               # KamisService 메인 클래스
├── types.py                 # 타입 정의 (ItemInfo, PriceInfo 등)
├── exceptions.py            # 예외 클래스
├── config.py                # 환경 설정 관리
├── cli.py                   # CLI 명령어 도구
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

### 4. 17개 KAMIS API 지원

- `daily_by_category`: 일별 부류별 가격
- `daily_by_item_period`: 일별 품목별 가격 (기간)
- `monthly_sales`: 월별 평균가격
- `yearly_sales`: 연별 평균가격
- `recent_price_trend`: 최근 가격 동향
- `region_item`: 지역별 품목별 가격
- 그 외 11개 API...

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

## 📄 라이선스

MIT License - 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.