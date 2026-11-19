"""Blue Book P-14.2.1.1/.2 and Table 1.4, through the hundreds.

The bounded vocabulary keeps the shared name recognizers practical. It is a
coverage limit, not a chemical maximum or an extension to polymer naming.
"""

MAX_CARBON_CHAIN_LENGTH = 999

_UNITS = ('', 'hen', 'do', 'tri', 'tetra', 'penta', 'hexa', 'hepta', 'octa', 'nona')
_TENS = ('', 'deca', 'icosa', 'triaconta', 'tetraconta', 'pentaconta',
         'hexaconta', 'heptaconta', 'octaconta', 'nonaconta')
_HUNDREDS = ('', 'hecta', 'dicta', 'tricta', 'tetracta', 'pentacta',
             'hexacta', 'heptacta', 'octacta', 'nonacta')


def basic_number_term(number: int) -> str:
    """Return a simple numerical term, or an empty string outside coverage.

    Composite terms cite units, tens, then hundreds. Eleven keeps undeca;
    one/two become hen/do in composites. The i of icosa is elided after a
    vowel. Meth/eth/prop/but are carbon-hydride roots, not numerical terms.
    """
    if not isinstance(number, int) or isinstance(number, bool) or not 1 <= number <= MAX_CARBON_CHAIN_LENGTH:
        return ''
    if number < 3:
        return ('mono', 'di')[number - 1]
    lower = number % 100
    if lower == 11:
        term = 'undeca'
    else:
        unit = _UNITS[lower % 10]
        tens = _TENS[lower // 10]
        if tens == 'icosa' and unit.endswith(('a', 'e', 'i', 'o', 'u')):
            tens = tens[1:]
        term = unit + tens
    return term + _HUNDREDS[number // 100]
