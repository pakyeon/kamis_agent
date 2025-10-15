# -*- coding: utf-8 -*-
"""텍스트 정규화 및 토큰화"""

import re
import unicodedata
from typing import List
from kiwipiepy import Kiwi


class TextProcessor:
    """텍스트 전처리 및 토큰화"""

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
        """
        if not text:
            return ""

        # 1. 유니코드 정규화
        text = unicodedata.normalize("NFKC", text).strip()

        # 2. 특수문자 제거
        text = self._SIMPLE_CLEAN_RE.sub(" ", text)
        text = self._WHITESPACE_RE.sub(" ", text).strip()

        # 3. 형태소 분석 및 불필요한 품사 제거
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
