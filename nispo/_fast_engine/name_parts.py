from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Prefix:
    text: str
    locants: tuple[str, ...] = ()

    def render(self) -> str:
        if not self.locants:
            return self.text
        return f"{','.join(self.locants)}-{self.text}"


@dataclass(frozen=True, slots=True)
class Suffix:
    text: str
    separator: str = "-"

    def render_after(self, name: str) -> str:
        return f"{name}{self.separator}{self.text}"


@dataclass(frozen=True, slots=True)
class NameParts:
    parent: str
    prefixes: tuple[Prefix, ...] = ()
    suffixes: tuple[Suffix, ...] = ()
    stereo: tuple[str, ...] = ()
    charge: str = ""

    def render(self) -> str:
        pieces = [prefix.render() for prefix in self.prefixes if prefix.text]
        if self.stereo:
            pieces.insert(0, f"({','.join(self.stereo)})")
        pieces.append(self.parent)
        name = "-".join(piece for piece in pieces if piece)
        for suffix in self.suffixes:
            if suffix.text:
                name = suffix.render_after(name)
        if self.charge:
            name = f"{name}{self.charge}"
        return name
