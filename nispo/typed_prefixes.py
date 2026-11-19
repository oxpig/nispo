"""Source-typed rendering for mapped, locanted substituent prefixes.

The caller supplies the mapped component and its independently established
simple/compound nature. This module does not infer chemistry from locants or
from the spelling of an unknown component.
"""

from collections import defaultdict
import re

from .formatting import enclose_name_unit, prefix_alphabetization_key
from .substituent_multiplication import (
    multiply_substituent,
    simple_component_has_internal_numeric_locants,
)


RULE = "P-14.5.1/P-14.5.4/P-16.3/P-16.5.1.2/P-16.5.1.4/P-16.5.4"


def typed_prefix_text(rows, *, citation_order=None):
    """Render ``(host_locant, component_text, nature)`` rows or return None.

    Names are already rendered by the mapped component producer. ``nature``
    must be its explicit ``simple`` or ``compound`` proof. Identical complete
    components are grouped before applying P-16 multiplication. Alphabetical
    order is determined from the unmultiplied component, with internal
    component locants participating before any host locant (P-14.5.4).

    Under the bounded P-16.5.1.2 rule, a singleton source-typed simple
    component is enclosed when its admitted name grammar proves internal
    numeric locants and the mapped host locant is numeric. Letter-host rows
    need parent-locant context not carried by this API and are not adjudicated
    here.

    The bounded API handles explicit locants only. It returns the empty string
    for no rows and declines unknown component grammars or malformed rows.
    """
    if not isinstance(rows, (list, tuple)):
        return None
    if not rows:
        if citation_order is None:
            return ""
        if (not isinstance(citation_order, (tuple, list))
                or any(not isinstance(name, str) or not name for name in citation_order)
                or len(citation_order) != len(set(citation_order))
                or citation_order):
            return None
        return ""
    grouped = defaultdict(list)
    nature_by_component = {}
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            return None
        locant, component, nature = row
        if (not isinstance(locant, str) or not locant or not isinstance(component, str)
                or not component or not _supported_host_locant(locant)
                or not isinstance(nature, str) or nature not in {"simple", "compound"}):
            return None
        prior = nature_by_component.get(component)
        if prior is not None and prior != nature:
            return None
        nature_by_component[component] = nature
        if multiply_substituent(component, 1, nature=nature) is None:
            return None
        grouped[component].append(locant)

    if citation_order is not None:
        if (not isinstance(citation_order, (tuple, list))
                or any(not isinstance(name, str) or not name for name in citation_order)
                or len(citation_order) != len(set(citation_order))
                or set(citation_order) != set(grouped)):
            return None
        ordered = tuple(citation_order)
    else:
        # P-14.5 ordering is applied to the component identity before the outer
        # multiplier is inserted. The key includes internal locants for a tie.
        ordered = sorted(grouped, key=prefix_alphabetization_key)
    rendered = []
    for component in ordered:
        locants = sorted(grouped[component], key=_host_locant_key)
        nature = nature_by_component[component]
        multiplied = multiply_substituent(component, len(locants), nature=nature)
        if multiplied is None:
            return None
        body = multiplied.text
        if (len(locants) == 1 and nature == "compound"):
            body = enclose_name_unit(component)
        elif (len(locants) == 1 and nature == "simple"
              and simple_component_has_internal_numeric_locants(component)
              and _numeric_host_locant(locants[0])):
            # P-16.5.1.2 is distinct from compound-prefix enclosure: the
            # validated internal locants and this mapped numeric host locant
            # refer to different structure elements. Keep the component
            # source-typed as simple and leave multiplication unchanged.
            body = enclose_name_unit(component)
        rendered.append(f"{','.join(locants)}-{body}")
    return "-".join(rendered)


def _host_locant_key(locant):
    """Sort admitted host locants with the shared P-14/P-16 source key."""
    from .locant_order import writing_locant_key

    return writing_locant_key(locant)


def _supported_host_locant(locant):
    """Admit numeric and source-ordered N-site parent-position locants."""
    from .locant_order import writing_locant_key

    bounded = (locant in {"N", "N'"}
               or re.fullmatch(r"[1-9]\d*[a-z']*", locant) is not None
               or re.fullmatch(r"N'*\^\([1-9]\d*'*(?:\^\d+)?\)", locant) is not None)
    return bounded and writing_locant_key(locant) is not None


def _numeric_host_locant(locant):
    """Recognize the numeric locant class, not N-site letter locants."""
    return re.fullmatch(r"[1-9]\d*[a-z']*", locant) is not None
