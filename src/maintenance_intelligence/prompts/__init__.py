"""Versioned prompt registry.

Prompts are plain Markdown files next to this module. :func:`load_prompt` returns
the text together with a content hash, and every eval run and trace records that
hash, so a metric can always be tied back to the exact prompt wording that produced
it. Editing a prompt therefore shows up as a new "prompt version" in MLflow and in
traces without any manual bookkeeping.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path


@dataclass(frozen=True)
class Prompt:
    name: str
    text: str
    sha256: str

    @property
    def version(self) -> str:
        return self.sha256[:12]

    def render(self, **values: str) -> str:
        rendered = self.text
        for key, value in values.items():
            rendered = rendered.replace("{{" + key + "}}", value)
        return rendered


def load_prompt(name: str) -> Prompt:
    package = resources.files(__name__)
    path = Path(str(package / f"{name}.md"))
    text = path.read_text(encoding="utf-8")
    return Prompt(name=name, text=text, sha256=hashlib.sha256(text.encode("utf-8")).hexdigest())


def list_prompts() -> dict[str, str]:
    """Map of prompt name -> short version, for /evaluation and MLflow params."""
    package = resources.files(__name__)
    versions: dict[str, str] = {}
    for entry in Path(str(package)).glob("*.md"):
        versions[entry.stem] = load_prompt(entry.stem).version
    return dict(sorted(versions.items()))
