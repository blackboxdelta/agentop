from __future__ import annotations

import pytest

from agentop.config import load_config, normalize_host


def test_normalize_host_adds_scheme_and_removes_trailing_slash():
    assert normalize_host("localhost:11434/") == "http://localhost:11434"
    assert normalize_host("https://remote.example/") == "https://remote.example"


def test_load_config_precedence_file_then_env_then_cli(tmp_path, monkeypatch):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
[ollama]
host = "file-host:11434"
[polling]
refresh_interval = 2
tags_interval = 45
request_timeout = 8
[state]
path = "~/custom-events.db"
"""
    )
    monkeypatch.setenv("OLLAMA_HOST", "env-host:11434")
    config = load_config(
        config_path=config_path,
        host="cli-host:11434",
        refresh_interval=0.5,
    )
    assert config.host == "http://cli-host:11434"
    assert config.refresh_interval == 0.5
    assert config.tags_interval == 45
    assert config.request_timeout == 8
    assert config.state_path.name == "custom-events.db"


def test_load_config_honors_no_color_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    config = load_config(config_path=tmp_path / "missing.toml")
    assert config.no_color is True


def test_invalid_host_and_intervals_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        normalize_host("ftp://bad.example")
    path = tmp_path / "config.toml"
    path.write_text("[polling]\nrefresh_interval = 0\n")
    with pytest.raises(ValueError):
        load_config(config_path=path)
