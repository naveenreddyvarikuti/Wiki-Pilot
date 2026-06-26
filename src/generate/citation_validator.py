import re
from dataclasses import dataclass


@dataclass
class CitationResult:
    cited_sources: set[int]
    invalid_sources: set[int]
    num_sources_provided: int
    num_citations: int
    has_citations: bool
    all_valid: bool

    @property
    def coverage(self) -> float:
        if self.num_sources_provided == 0:
            return 0.0
        return len(self.cited_sources) / self.num_sources_provided

    @property
    def is_valid(self) -> bool:
        return self.has_citations and self.all_valid


def validate_citations(answer: str, num_sources: int) -> CitationResult:
    matches = re.findall(r'\[Source\s+(\d+)\]', answer)

    cited = set()
    invalid = set()

    for m in matches:
        n = int(m)
        if 1 <= n <= num_sources:
            cited.add(n)
        else:
            invalid.add(n)

    return CitationResult(
        cited_sources=cited,
        invalid_sources=invalid,
        num_sources_provided=num_sources,
        num_citations=len(matches),
        has_citations=len(matches) > 0,
        all_valid=len(invalid) == 0,
    )


def format_validation_report(result: CitationResult) -> str:
    lines = []

    status = "PASS" if result.is_valid else "FAIL"
    lines.append(f"Citation check: {status}")

    if not result.has_citations:
        lines.append("  No citations found in answer")
        return "\n".join(lines)

    lines.append(f"  Citations found: {result.num_citations}")
    lines.append(f"  Sources cited: {sorted(result.cited_sources)}")
    lines.append(f"  Coverage: {result.coverage:.0%} ({len(result.cited_sources)}/{result.num_sources_provided} sources used)")

    if result.invalid_sources:
        lines.append(f"  Invalid references: {sorted(result.invalid_sources)}")


    return "\n".join(lines)
