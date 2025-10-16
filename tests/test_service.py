# -*- coding: utf-8 -*-
"""KAMIS Service 간단 테스트

실행 방법:
    프로젝트 루트(kamis_agent/)의 tests 폴더에서:
    python test_service.py
"""

import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# 프로젝트 루트의 상위 디렉토리를 sys.path에 추가
project_parent = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_parent))

# .env 파일 명시적으로 로드 (프로젝트 루트에서)
project_root = Path(__file__).parent.parent
env_path = project_root / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✓ .env 파일 로드: {env_path}")
else:
    print(f"⚠ .env 파일 없음: {env_path}")

# 환경변수 확인
print(f"✓ KAMIS_CERT_KEY: {'설정됨' if os.getenv('KAMIS_CERT_KEY') else '없음'}")
print(f"✓ OPENAI_API_KEY: {'설정됨' if os.getenv('OPENAI_API_KEY') else '없음'}")

# 이제 kamis_agent를 패키지로 import
from kamis_agent import KamisService


def print_separator(title: str):
    """구분선 출력"""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60 + "\n")


def test_search():
    """search() 테스트"""
    print_separator("search() 테스트")

    kamis = KamisService()

    query = "사과 가격"
    print(f"쿼리: {query}\n")

    result = kamis.search(query)

    print(json.dumps(result, indent=2, ensure_ascii=False))


def test_answer():
    """answer() 테스트"""
    print_separator("answer() 테스트")

    kamis = KamisService()

    query = "오늘 사과 가격은?"
    print(f"쿼리: {query}\n")

    answer = kamis.answer(query)

    print(f"답변:\n{answer}")


def main():
    """메인 함수"""
    try:
        test_search()
        test_answer()

        print_separator("테스트 완료")

    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
