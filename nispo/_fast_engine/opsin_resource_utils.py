from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from xml.etree import ElementTree as ET


TOKEN_RESOURCE_DIR = ("resources", "opsin_tokens")


@dataclass(frozen=True)
class OpsinTokenListContext:
    """OPSIN token-list metadata used to keep inverse rules grouped."""

    tagname: str
    symbol: str
    type: str = ""
    sub_type: str = ""


@dataclass(frozen=True)
class OpsinTokenRecord:
    """One token plus the OPSIN resource subset it came from."""

    file_name: str
    context: OpsinTokenListContext
    token: ET.Element


def iter_opsin_token_records(file_names: tuple[str, ...]):
    for file_name in file_names:
        xml_text = read_opsin_resource(file_name)
        if not xml_text:
            continue
        root = ET.fromstring(xml_text)
        for token_list in root.findall(".//tokenList"):
            context = OpsinTokenListContext(
                tagname=token_list.attrib.get("tagname", ""),
                symbol=token_list.attrib.get("symbol", ""),
                type=token_list.attrib.get("type", ""),
                sub_type=token_list.attrib.get("subType", ""),
            )
            for token in token_list.findall("./token"):
                yield OpsinTokenRecord(file_name=file_name, context=context, token=token)


def iter_opsin_tokens(file_names: tuple[str, ...]):
    for record in iter_opsin_token_records(file_names):
        yield record.token


def read_opsin_resource(file_name: str) -> str:
    try:
        path = resources.files(__package__)
        for part in TOKEN_RESOURCE_DIR:
            path = path.joinpath(part)
        return path.joinpath(file_name).read_text(encoding="utf-8")
    except Exception:
        return ""


def preferred_token_name(text: str, normalizations: dict[str, str] | None = None) -> str:
    aliases = [alias.strip() for alias in text.split("|") if alias.strip()]
    if not aliases:
        return ""
    normalizations = normalizations or {}
    for alias in aliases:
        if " " not in alias and alias.isascii():
            return normalizations.get(alias, alias)
    return normalizations.get(aliases[0], aliases[0]) if aliases[0].isascii() else ""


def primary_label(label: str) -> str:
    return label.split(",", 1)[0]


def unsupported_token_attributes(token, supported: set[str]) -> list[str]:
    return [key for key in token.attrib if key not in supported]
