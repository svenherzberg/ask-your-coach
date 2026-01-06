from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any

try:
    import yaml
except Exception:
    yaml = None


@dataclass
class AppConfig:
    # LLM endpoints
    lmstudio_url: Optional[str] = None
    lmstudio_model: Optional[str] = None
    ayc_ll_model: Optional[str] = None

    # TTS / audio
    tts_sample_rate: int = 22050
    tts_player: Optional[str] = None

    # Prompt / mode
    default_mode: str = "default"

    # Low-level options
    n_threads: int = 8

    # raw config loaded from file (if any)
    raw: Dict[str, Any] | None = None


_GLOBAL_CONFIG: Optional[AppConfig] = None


def _load_yaml_config(paths: list[Path]) -> Dict[str, Any]:
    for p in paths:
        if p.exists() and yaml is not None:
            try:
                return yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
    return {}


def load_config(config_path: Optional[str] = None) -> AppConfig:
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is not None:
        return _GLOBAL_CONFIG

    # env first
    lmstudio_url = os.environ.get("LMSTUDIO_URL")
    lmstudio_model = os.environ.get("LMSTUDIO_MODEL")
    ayc_ll_model = os.environ.get("AYC_LL_MODEL")
    tts_sample_rate = int(os.environ.get("AYC_TTS_SAMPLE_RATE", "22050"))
    tts_player = os.environ.get("AYC_TTS_PLAYER")
    default_mode = os.environ.get("AYC_DEFAULT_MODE", "default")
    n_threads = int(os.environ.get("AYC_N_THREADS", str(os.cpu_count() or 4)))

    # file config (project config.yaml or user config)
    cfg_paths = []
    if config_path:
        cfg_paths.append(Path(config_path))
    # project config in repo root `config.yaml`
    repo_cfg = Path(__file__).parent.parent / "config.yaml"
    cfg_paths.append(repo_cfg)
    # user config
    cfg_paths.append(Path.home() / ".config" / "ask-your-coach" / "config.yaml")

    file_cfg = _load_yaml_config(cfg_paths)

    # combine: file config overrides env if keys present
    if file_cfg:
        lmstudio_url = file_cfg.get("lmstudio_url", lmstudio_url)
        lmstudio_model = file_cfg.get("lmstudio_model", lmstudio_model)
        ayc_ll_model = file_cfg.get("ayc_ll_model", ayc_ll_model)
        tts_sample_rate = int(file_cfg.get("tts_sample_rate", tts_sample_rate))
        tts_player = file_cfg.get("tts_player", tts_player)
        default_mode = file_cfg.get("default_mode", default_mode)
        n_threads = int(file_cfg.get("n_threads", n_threads))

    cfg = AppConfig(
        lmstudio_url=lmstudio_url,
        lmstudio_model=lmstudio_model,
        ayc_ll_model=ayc_ll_model,
        tts_sample_rate=tts_sample_rate,
        tts_player=tts_player,
        default_mode=default_mode,
        n_threads=n_threads,
        raw=file_cfg or None,
    )
    _GLOBAL_CONFIG = cfg
    return cfg


def get_config() -> AppConfig:
    global _GLOBAL_CONFIG
    if _GLOBAL_CONFIG is None:
        return load_config()
    return _GLOBAL_CONFIG
