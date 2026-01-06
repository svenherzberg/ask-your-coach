from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except Exception:  # pragma: no cover - YAML optional
    yaml = None


class PromptManager:
    """Loads prompt templates and a config.yaml from the local `prompts` folder.

    Behavior:
    - Loads optional `config.yaml` in the prompts directory containing metadata per mode.
    - Templates remain simple `{}` format text files (e.g. `time_management.txt`).
    - `config.yaml` entries may provide `system` (inline template), `template_file`, and `generation` options.
    """

    def __init__(self, templates_dir: str | None = None):
        if templates_dir:
            self.templates_dir = Path(templates_dir)
        else:
            self.templates_dir = Path(__file__).parent

        self._config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        cfg_path = self.templates_dir / "config.yaml"
        if cfg_path.exists() and yaml is not None:
            try:
                return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            except Exception:
                return {}
        return {}

    def reload(self) -> None:
        """Reload templates/config from disk immediately."""
        self._config = self._load_config()

    async def watch_async(self, interval: float = 1.0, on_change: Optional[callable] = None) -> None:
        """Asynchronously watches the prompts directory for changes.

        When a change is detected, `reload()` is called and optional `on_change()` invoked.
        This coroutine runs until cancelled.
        """
        # collect mtimes of files in templates_dir
        def snapshot_mtimes():
            mt = {}
            for p in self.templates_dir.glob("**/*"):
                try:
                    mt[str(p)] = p.stat().st_mtime
                except Exception:
                    mt[str(p)] = None
            return mt

        last = snapshot_mtimes()
        try:
            while True:
                await asyncio.sleep(interval)
                cur = snapshot_mtimes()
                if cur != last:
                    # change detected
                    self.reload()
                    if on_change:
                        try:
                            on_change()
                        except Exception:
                            pass
                    last = cur
        except asyncio.CancelledError:
            return

    def _template_path(self, name: str) -> Path:
        return self.templates_dir / f"{name}.txt"

    def list_templates(self) -> list[str]:
        return [p.stem for p in self.templates_dir.glob("*.txt")]

    def render(self, name: str = "default", **kwargs: Any) -> str:
        # priority: config.system -> template_file in config -> {name}.txt
        entry = self._config.get(name, {}) if isinstance(self._config, dict) else {}
        if isinstance(entry, dict) and entry.get("system"):
            template = entry.get("system")
        else:
            template_file = (self.templates_dir / (entry.get("template_file") if isinstance(entry, dict) else "")) if entry else None
            if template_file and template_file.exists():
                template = template_file.read_text(encoding="utf-8")
            else:
                p = self._template_path(name)
                if not p.exists():
                    p = self._template_path("default")
                    if not p.exists():
                        raise FileNotFoundError(f"No template found for {name} and no default template present at {p}")
                template = p.read_text(encoding="utf-8")

        # Safe formatting: leave unknown placeholders intact
        class SafeDict(dict):
            def __missing__(self, key):
                return "{" + key + "}"

        return template.format_map(SafeDict(**kwargs))

    def get_generation_options(self, name: str = "default") -> Dict[str, Any]:
        entry = self._config.get(name, {}) if isinstance(self._config, dict) else {}
        gen = {}
        if isinstance(entry, dict):
            gen = entry.get("generation", {}) or {}
        # merge with top-level default generation options if present
        top = self._config.get("default_generation", {}) if isinstance(self._config, dict) else {}
        merged = dict(top)
        merged.update(gen)
        return merged
