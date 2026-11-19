"""Source-bounded typed locant ordering from Blue Book P-14.3.5/P-16.9.

The parser recognizes only locant forms whose order is explicitly supported
here. ``compare_locants`` and ``compare_locant_sets`` return ``None`` when a
form or cross-type comparison has no source-backed generic order. Rule-specific
criteria (for example, P-31's compound-locant criteria) belong to callers.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


_PRIMES = str.maketrans({"′": "'", "’": "'"})
_GREEK = {
    "α": 1, "β": 2, "γ": 3, "δ": 4, "ε": 5, "ζ": 6, "η": 7,
    "θ": 8, "ι": 9, "κ": 10, "λ": 11, "μ": 12, "ν": 13,
    "ξ": 14, "ο": 15, "π": 16, "ρ": 17, "σ": 18, "ς": 18,
    "τ": 19, "υ": 20, "φ": 21, "χ": 22, "ψ": 23, "ω": 24,
}


@dataclass(frozen=True, slots=True)
class Locant:
    """Parsed locant; `kind` is numeric, roman, greek, element, or compound."""

    kind: str
    primary: int | str
    prime_count: int = 0
    letter: str = ""
    superscript: int | None = None
    second: int | None = None


def parse_locant(value: object) -> Locant | None:
    """Parse a source-supported token; malformed or unproved syntax declines."""
    if not isinstance(value, str) or not value or value.strip() != value:
        return None
    text = value.translate(_PRIMES)

    # The second endpoint makes this a compound locant type, not punctuation
    # decorating a simple numeral. Ordering is handled by the caller's rule.
    match = re.fullmatch(r"(\d+)\((\d+)\)", text)
    if match:
        return Locant("compound", int(match[1]), second=int(match[2]))

    # P-16.9 element-site locants. P-66.4.1.4.1 establishes parent-position
    # order across N/N-prime sites; prime count resolves a tied parent site.
    match = re.fullmatch(r"(Se|Te|As|Sb|Bi|[A-Z])('*)\^(\d+)", text)
    if match:
        if match[1] not in {"N", "O", "P", "S", "Se", "Te", "As", "Sb", "Bi"}:
            return None
        return Locant("element", match[1], len(match[2]), superscript=int(match[3]))
    match = re.fullmatch(r"(Se|Te|As|Sb|Bi|[A-Z])('*)", text)
    if match:
        if match[1] not in {"N", "O", "P", "S", "Se", "Te", "As", "Sb", "Bi"}:
            return None
        return Locant("element", match[1], len(match[2]))

    if text in _GREEK:
        return Locant("greek", _GREEK[text])
    if re.fullmatch(r"[A-Z]", text) or re.fullmatch(r"[a-z]", text):
        return Locant("roman", text)

    # P-14.3.5 examples include 4′a (prime before letter), not 4a′.
    # Optional superscript is accepted after the whole number/letter token.
    match = re.fullmatch(r"(\d+)(\'*)([a-z]?)(?:\^(\d+))?", text)
    if not match:
        return None
    return Locant("numeric", int(match[1]), len(match[2]), match[3],
                  int(match[4]) if match[4] else None)


def n_site_key(position: int, prime_count: int = 0) -> tuple[int, int] | None:
    """Return the source-ordered key for an N site: parent position, then prime."""
    if (type(position) is not int or position < 1
            or type(prime_count) is not int or prime_count < 0):
        return None
    return (position, prime_count)


def n_site_sort_key(value: object) -> tuple[tuple[int, str], ...] | None:
    """Formatter key for the admitted N^n/N′^n forms only.

    The result retains this older adapter's one-entry tuple layout. New
    presentation callers use ``writing_locant_key`` for composite hosts too.
    """
    if not isinstance(value, str):
        return None
    text = value.replace("′", "'").replace("’", "'")
    match = re.fullmatch(r"N('*)\^(\d+)", text)
    if not match:
        return None
    position = int(match[2])
    if position < 1:
        return None
    return ((position, "N" + match[1]),)


def _n_parent_site(value):
    """Separate the N prime from a numeric/primed/composite parent locant."""
    if not isinstance(value, str):
        return None
    text = re.sub(r"\^\{(\d+)\}", r"^\1", value.translate(_PRIMES))
    match = re.fullmatch(r"N('*)(?:\^\((\d+'*(?:\^\d+)?)\)|\^(\d+)|(\d+'*(?:\^\d+)?))", text)
    if match is None:
        bare = re.fullmatch(r"N('*)", text)
        return (len(bare[1]), None) if bare is not None else None
    host = parse_locant(match[2] or match[3] or match[4])
    if host is None or host.kind != 'numeric' or host.primary < 1:
        return None
    return len(match[1]), host


def nitrogen_site_display(value):
    """P-16.9/P-66: the whole parent locant qualifies its N site.

    Parent-occurrence primes and ring-assembly composite indices stay inside
    that qualifier; they never become primes on the element symbol itself.
    """
    parsed = _n_parent_site(value)
    if parsed is None or parsed[1] is None:
        return value
    primes, host = parsed
    parent = str(host.primary) + "'" * host.prime_count
    if host.superscript is not None:
        parent += '^' + str(host.superscript)
    # Delimit even a simple numerical superscript in the plain-text form.
    # This is serialization, not a new chemical locant or ordering rule.
    return 'N' + "'" * primes + '^(' + parent + ')'


def writing_locant_key(value):
    """Source-ordered presentation key, including composite N qualifiers.

    Unlike a chemical preference comparison, a presentation caller must
    also retain unknown legacy text. Unknown forms receive no source key.
    """
    if not isinstance(value, str):
        return None
    n_site = _n_parent_site(value)
    if n_site is not None:
        primes, host = n_site
        return (0, 'N', (-1,) if host is None else _locant_key(host), primes)
    text = re.sub(r"\^\{(\d+)\}", r"^\1", value.translate(_PRIMES))
    parsed = parse_locant(text)
    if parsed is None or parsed.kind == 'compound':
        return None
    return _locant_key(parsed)


def _locant_key(locant: Locant):
    if locant.kind == "roman":
        # P-14.3.5: italic Roman letter locants precede Greek and numerals.
        return (0, locant.primary)
    if locant.kind == "greek":
        return (1, locant.primary)
    if locant.kind == "numeric":
        # Compare the leading locant first; at an identical number, the cited
        # examples put a primed numeric before a letter locant, then a
        # superscripted locant after the corresponding unsuperscripted form.
        if locant.superscript is not None:
            tail = (3, locant.superscript)
        elif locant.letter:
            tail = (2, locant.letter, locant.prime_count)
        elif locant.prime_count:
            tail = (1, locant.prime_count)
        else:
            tail = (0,)
        return (2, locant.primary, tail)
    if locant.kind == "element":
        # P-16.9.3 supports N before N′ for the same superscripted parent site.
        return (0, locant.primary,
                -1 if locant.superscript is None else locant.superscript,
                locant.prime_count)
    return None


def compare_locants(left: object, right: object) -> int | None:
    """Return -1/0/+1 if source establishes order; otherwise return None."""
    a, b = parse_locant(left), parse_locant(right)
    if a is None or b is None:
        return None
    if a.kind == "compound" or b.kind == "compound":
        return None
    if a.kind == "element" and b.kind == "element":
        if a.primary != b.primary:
            return None
        if (a.superscript is None) != (b.superscript is None):
            return None
    ka, kb = _locant_key(a), _locant_key(b)
    if ka is None or kb is None:
        return None
    return (ka > kb) - (ka < kb)


def compare_locant_sets(left, right) -> int | None:
    """Compare already-derived locant sets term-by-term; ``None`` is a barrier."""
    try:
        a, b = tuple(left), tuple(right)
    except TypeError:
        return None
    if any(parse_locant(value) is None for value in (*a, *b)):
        return None
    # A compound token needs a caller-owned rule even inside a set.
    if any(parse_locant(value).kind == "compound" for value in (*a, *b)):
        return None
    # Sort via pairwise source comparator; any incomparable pair is unknown.
    def ordered(values):
        out = []
        for value in values:
            placed = False
            for index, prior in enumerate(out):
                comparison = compare_locants(value, prior)
                if comparison is None:
                    return None
                if comparison < 0:
                    out.insert(index, value)
                    placed = True
                    break
            if not placed:
                out.append(value)
        return tuple(out)
    a_sorted, b_sorted = ordered(a), ordered(b)
    if a_sorted is None or b_sorted is None:
        return None
    for x, y in zip(a_sorted, b_sorted):
        comparison = compare_locants(x, y)
        if comparison is None or comparison:
            return comparison
    return (len(a_sorted) > len(b_sorted)) - (len(a_sorted) < len(b_sorted))
