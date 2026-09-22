from __future__ import annotations
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, Context

_NUMBER_TOKEN_RE = re.compile(
    r"[-+]?\d{1,3}(?:,\d{3})+(?:\.\d+)?"
    r"|[-+]?\d+(?:\.\d+)?%"
    r"|[-+]?\d+\.\d+"
    r"|[-+]?\d+"
)

# A token that starts with a letter and carries digits anywhere inside its
# unbroken run is an identifier, not a fact number: station codes ("J9", "J12"),
# query ids ("S001"), and cited chunk ids ("kim2024_hakashiwo::0017"). Every
# digit in that run belongs to the identifier, not just the one touching the
# letter, so the whole span is masked before numbers are extracted.
_IDENTIFIER_RE = re.compile(r"[A-Za-z가-힣][A-Za-z가-힣0-9_:\-]*\d[A-Za-z가-힣0-9_:\-]*")


# Ordered-list markers in a Markdown-formatted answer ("1.", "**2.**", "3)") are
# document structure, not reported facts.
_LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[*\-+][ \t]*)?(?:\*\*)?(\d+)[.)]", re.MULTILINE)


def _identifier_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in _IDENTIFIER_RE.finditer(text)]


def _list_marker_spans(text: str) -> list[tuple[int, int]]:
    return [m.span(1) for m in _LIST_MARKER_RE.finditer(text)]


def _overlaps(start: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(start < s_end and s_start < end for s_start, s_end in spans)


def _is_excluded_span(text: str, start: int, end: int) -> bool:
    suffix = text[end:end + 3]
    if re.match(r"\s*(절|장|조)\b", suffix):
        return True
    # Calendar labels echoed from the query ("4월", "3분기") are not computed
    # facts and must not be scored as reported numbers.
    if suffix.lstrip().startswith(("월", "분기")):
        return True
    before = text[max(0, start - 4):start]
    if re.search(r"(표|그림)\s*$", before):
        return True
    return False


def extract_numbers(text: str) -> list[Decimal]:
    skip_spans = _identifier_spans(text) + _list_marker_spans(text)
    results: list[Decimal] = []
    for m in _NUMBER_TOKEN_RE.finditer(text):
        raw = m.group(0)
        if _overlaps(m.start(), m.end(), skip_spans):
            continue
        if _is_excluded_span(text, m.start(), m.end()):
            continue
        cleaned = raw.rstrip("%").replace(",", "")
        try:
            value = Decimal(cleaned)
        except InvalidOperation:
            continue
        results.append(value)
    return results


def flatten_facts(value) -> list[Decimal]:
    out: list[Decimal] = []
    if isinstance(value, dict):
        for v in value.values():
            out.extend(flatten_facts(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(flatten_facts(v))
    elif isinstance(value, bool):
        return out
    elif isinstance(value, (int, float)):
        out.append(Decimal(str(value)))
    elif isinstance(value, str):
        try:
            out.append(Decimal(value))
        except InvalidOperation:
            pass
    return out


@dataclass
class ComparisonResult:
    matched_exact: list = field(default_factory=list)
    matched_rounded: list = field(default_factory=list)
    unmatched: list = field(default_factory=list)
    pass_: bool = False


def _same_sigfig(a: Decimal, b: Decimal, sigfig: int) -> bool:
    if a == 0 or b == 0:
        return a == b
    ctx = Context(prec=sigfig)
    return ctx.create_decimal(a) == ctx.create_decimal(b)


def compare(reported: list[Decimal], facts: list[Decimal], policy: str, sigfig: int = 6) -> ComparisonResult:
    result = ComparisonResult()
    fact_exact = set(facts)
    fact_rounded = {f.quantize(Decimal("1")) for f in facts}
    for num in reported:
        if num in fact_exact:
            result.matched_exact.append(num)
            continue
        if policy == "round" and num.quantize(Decimal("1")) in fact_rounded:
            result.matched_rounded.append(num)
            continue
        if policy == "sigfig" and any(_same_sigfig(num, f, sigfig) for f in facts):
            result.matched_rounded.append(num)
            continue
        result.unmatched.append(num)
    result.pass_ = len(result.unmatched) == 0
    return result
