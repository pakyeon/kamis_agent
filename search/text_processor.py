# -*- coding: utf-8 -*-
"""텍스트 정규화 및 토큰화"""

import re
import unicodedata
from typing import List
from kiwipiepy import Kiwi


class TextProcessor:
    """
    한국어 텍스트 정규화

    유니코드 정규화만으로는 한국어 검색 매칭이 어려운 문제를 해결하기 위해
    형태소 분석기(kiwipiepy)를 사용하여 조사/어미/기호를 제거합니다.

    사용처:
        - data/transformer.py: DB 적재 전 품목명/품종명 정규화
        - search/searcher.py: LLM이 추출한 키워드 정규화
    """

    # 정규표현식 패턴 (컴파일하여 재사용)
    _SIMPLE_CLEAN_RE = re.compile(r"[^\w\s가-힣]")
    _WHITESPACE_RE = re.compile(r"\s+")
    _DIGIT_RE = re.compile(r"\d+")

    def __init__(self):
        self.kiwi = Kiwi()

    def normalize(self, text: str) -> str:
        """
        텍스트 정규화 및 토큰화

        Args:
            text: 원본 텍스트

        Returns:
            정규화된 텍스트 (공백으로 구분된 토큰)

        처리 과정:
            1. 유니코드 정규화 (NFKC)
            2. 특수문자 제거
            3. 형태소 분석으로 조사(J), 어미(E), 기호(S) 제거
        """
        if not text:
            return ""

        # 1. 유니코드 정규화
        text = unicodedata.normalize("NFKC", text).strip()

        # 2. 특수문자 제거
        text = self._SIMPLE_CLEAN_RE.sub(" ", text)
        text = self._WHITESPACE_RE.sub(" ", text).strip()

        # 3. 형태소 분석 및 불필요한 품사 제거
        # J (조사): ~의, ~를, ~에게
        # E (어미): ~ㅂ니다, ~었다
        # S (기호): !, ?, ,
        tokens = [
            t.form
            for t in self.kiwi.tokenize(text)
            if not (
                t.tag.startswith("J") or t.tag.startswith("E") or t.tag.startswith("S")
            )
        ]

        return "".join(tokens)

    def extract_digits(self, text: str) -> List[str]:
        """텍스트에서 숫자 추출"""
        return self._DIGIT_RE.findall(text)
