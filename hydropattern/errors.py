'''Canonical error envelope definitions for hydropattern parser/service failures.'''

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ParserErrorCode(StrEnum):
    '''Stable parser/service error codes.'''

    MISSING_SECTION = 'PARSER_MISSING_SECTION'
    MISSING_FIELD = 'PARSER_MISSING_FIELD'
    INVALID_TYPE = 'PARSER_INVALID_TYPE'
    INVALID_VALUE = 'PARSER_INVALID_VALUE'
    UNKNOWN_CHARACTERISTIC = 'PARSER_UNKNOWN_CHARACTERISTIC'
    UNKNOWN_COMPARISON_SYMBOL = 'PARSER_UNKNOWN_COMPARISON_SYMBOL'


@dataclass(frozen=True)
class CanonicalErrorEnvelope:
    '''Machine-readable canonical error payload.'''

    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    source: str = 'parser'

    def as_dict(self) -> dict[str, Any]:
        '''Return the envelope as a plain dictionary.'''
        return {
            'code': self.code,
            'message': self.message,
            'context': self.context,
            'source': self.source,
        }


class CanonicalHydropatternError(ValueError):
    '''ValueError subclass that carries a canonical error envelope.'''

    def __init__(self, envelope: CanonicalErrorEnvelope):
        super().__init__(envelope.message)
        self.envelope = envelope


def raise_parser_error(code: ParserErrorCode | str, message: str, **context: Any) -> None:
    '''Raise a canonical parser error with stable code and machine-readable context.'''
    raise CanonicalHydropatternError(
        CanonicalErrorEnvelope(code=str(code), message=message, context=context),
    )