from __future__ import annotations


def template_record_name(template: dict) -> str:
    return template["name"]


def template_record_mol(template: dict):
    return template["mol"]


def template_record_locants(template: dict) -> dict:
    return template["locants"]
