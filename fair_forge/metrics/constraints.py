"""Constraint protocol and built-in implementations for ICR computation."""

import json
import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class Constraint(Protocol):
    """Verifies a single programmatic constraint on a model response."""

    def check(self, response: str) -> bool: ...


class JsonConstraint:
    """Passes when the response is valid JSON."""

    def check(self, response: str) -> bool:
        try:
            json.loads(response)
            return True
        except (json.JSONDecodeError, ValueError):
            return False


class WordCountConstraint:
    """Passes when the response contains at most max_words words."""

    def __init__(self, max_words: int):
        self._max = max_words

    def check(self, response: str) -> bool:
        return len(response.split()) <= self._max


class KeywordConstraint:
    """Passes when the response contains the required keyword."""

    def __init__(self, keyword: str, case_sensitive: bool = False):
        self._keyword = keyword
        self._case_sensitive = case_sensitive

    def check(self, response: str) -> bool:
        if self._case_sensitive:
            return self._keyword in response
        return self._keyword.lower() in response.lower()


class RegexConstraint:
    """Passes when the response matches the given regular expression."""

    def __init__(self, pattern: str):
        self._pattern = re.compile(pattern)

    def check(self, response: str) -> bool:
        return bool(self._pattern.search(response))
