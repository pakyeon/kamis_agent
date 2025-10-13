# -*- coding: utf-8 -*-
"""KAMIS Service CLI"""

import sys
import argparse
import json
import logging

from service import KamisService

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(
        description="KAMIS 농축수산물 가격 정보 서비스 CLI"
    )

    subparsers = parser.add_subparsers(dest="command", help="명령어")

    # query 명령
    query_parser = subparsers.add_parser("query", help="질의 (자연어 답변)")
    query_parser.add_argument("text", help="자연어 질문")
    query_parser.add_argument("--raw", action="store_true", help="구조화된 데이터 반환")

    # search 명령어
    search_parser = subparsers.add_parser("search", help="품목 검색")
    search_parser.add_argument("item", help="품목명")
    search_parser.add_argument("--top-k", type=int, default=5, help="결과 개수")

    # update 명령어
    update_parser = subparsers.add_parser("update", help="DB 업데이트")
    update_parser.add_argument("--force", action="store_true", help="강제 업데이트")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        # 서비스 초기화
        service = KamisService()

        # 명령어 실행
        if args.command == "query":
            handle_query(service, args)
        elif args.command == "search":
            handle_search(service, args)
        elif args.command == "update":
            handle_update(service, args)

    except Exception as e:
        logger.error(f"오류 발생: {e}")
        sys.exit(1)


def handle_query(service: KamisService, args):
    """통합 질의"""
    print(f"\n💬 질문: {args.text}\n")

    if args.raw:
        # 구조화된 데이터
        result = service.search(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        # 자연어 답변
        answer = service.answer(args.text)
        print(f"답변: {answer}\n")


def handle_search(service: KamisService, args):
    """품목 검색"""
    result = service.search_item(args.item, top_k=args.top_k)

    if not result:
        print(f"❌ 품목을 찾을 수 없습니다: {args.item}")
        return

    print(f"\n🔍 검색 결과: {args.item}\n")

    results = result if isinstance(result, list) else [result]
    for i, item in enumerate(results, 1):
        print(f"{i}. {item['product']['name']} ({item['product']['code']})")
        print(f"   부류: {item['category']['name']}")
        if "kind" in item:
            print(f"   품종: {item['kind']['name']}")
        if "grade" in item:
            print(f"   등급: {item['grade'].get('name', 'N/A')}")
        print()


def handle_update(service: KamisService, args):
    """DB 업데이트"""
    print("\n🔄 DB 업데이트 중...\n")

    if args.force:
        from data import DataManager

        manager = DataManager(service.config.db_path)
        manager.update(auto_download=True)
        print("✅ 강제 업데이트 완료")
    else:
        # 자동으로 TTL 체크
        print("✅ DB 상태 확인 완료 (필요시 자동 업데이트)")


if __name__ == "__main__":
    main()
