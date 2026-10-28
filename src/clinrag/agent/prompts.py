"""Versioned prompt templates (prompts/v*.yaml)."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import yaml

REQUIRED_FIELDS = ("version", "name", "system", "user_template")


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    name: str
    system: str
    user_template: str
    no_context_note: str = "(no excerpts retrieved)"
    description: str = ""
    changes: str = ""
    path: str = ""

    @property
    def sha(self) -> str:
        """Short content hash, logged with every eval run so results map to exact text."""
        return hashlib.sha256((self.system + self.user_template).encode()).hexdigest()[:10]

    def render_user(self, question: str, context: str) -> str:
        return self.user_template.format(question=question, context=context or self.no_context_note)


def _prompt_path(prompts_dir: Path, version: str) -> Path:
    exact = prompts_dir / f"{version}.yaml"
    if exact.exists():
        return exact
    matches = sorted(prompts_dir.glob(f"{version}_*.yaml"))
    if not matches:
        available = ", ".join(p.version for p in list_prompts(prompts_dir)) or "none"
        raise FileNotFoundError(f"No prompt {version!r} in {prompts_dir} (available: {available})")
    return matches[0]


def load_prompt(prompts_dir: Path, version: str) -> PromptTemplate:
    path = _prompt_path(prompts_dir, version)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [f for f in REQUIRED_FIELDS if not data.get(f)]
    if missing:
        raise ValueError(f"{path.name} is missing fields: {missing}")
    if "{question}" not in data["user_template"] or "{context}" not in data["user_template"]:
        raise ValueError(f"{path.name}: user_template needs {{question}} and {{context}}")
    known = {f for f in PromptTemplate.__dataclass_fields__ if f != "path"}
    return PromptTemplate(**{k: v for k, v in data.items() if k in known}, path=str(path))


def list_prompts(prompts_dir: Path) -> list[PromptTemplate]:
    out = []
    for path in sorted(prompts_dir.glob("v*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if data.get("version"):
            out.append(load_prompt(prompts_dir, data["version"]))
    return out
