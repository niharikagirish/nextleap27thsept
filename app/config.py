"""Typed settings from config.yaml + .env.

Secrets are read from environment variables only, never from config.yaml, so
config.yaml stays safe to commit and safe to display during a demo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    config: dict[str, Any]
    root: Path

    def get(self, dotted: str, default: Any = None) -> Any:
        """Look up a nested key by dotted path.

        >>> SETTINGS.get("retrieval.score_floor")
        0.35
        """
        node: Any = self.config
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str, default: str) -> Path:
        """Resolve a configured relative path against the project root."""
        return self.root / str(self.get(dotted, default))

    @property
    def llm_api_key(self) -> str | None:
        env_name = self.get("generation.api_key_env", "LLM_API_KEY")
        value = os.getenv(env_name)
        return value or None


def load_settings(path: Path | None = None) -> Settings:
    cfg_path = path or (ROOT / "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    return Settings(config=cfg, root=ROOT)


SETTINGS = load_settings()
