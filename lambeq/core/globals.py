from __future__ import annotations

from enum import Enum


class VerbosityLevel(Enum):
    PROGRESS = 'progress'
    SUPPRESS = 'suppress'
    TEXT = 'text'

    @classmethod
    def has_value(self, value: str):
        return value in [c.value for c in self]
