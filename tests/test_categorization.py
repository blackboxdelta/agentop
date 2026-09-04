"""Unit tests for the process classification engine.

Pure function, fabricated inputs — no real OS processes involved. Covers
every signature observed in the wild plus the negative case (unrelated
macOS system agents must NOT be swept up by a generic "agent" match).
"""
from agentop.collectors.processes import classify
from agentop.models import Category, Risk


def test_ollama_server():
    r = classify(
        "/Applications/Ollama.app/Contents/Resources/ollama",
        ["ollama", "serve"],
        "ollama",
    )
    assert r.category is Category.MODEL_SERVER
    assert r.subtype == "ollama-server"
    assert r.risk is Risk.MEDIUM


def test_ollama_tray_app():
    r = classify("/Applications/Ollama.app/Contents/MacOS/Ollama", ["Ollama"], "Ollama")
    assert r.category is Category.MODEL_SERVER
    assert r.subtype == "ollama-app"


def test_windows_ollama_server_and_worker_are_classified():
    server = classify(
        r"C:\Users\user\AppData\Local\Programs\Ollama\ollama.exe",
        ["ollama.exe", "serve"],
        "ollama.exe",
    )
    worker = classify(
        r"C:\Users\user\AppData\Local\Programs\Ollama\ollama_llama_server.exe",
        ["ollama_llama_server.exe"],
        "ollama_llama_server.exe",
    )
    assert server.category is Category.MODEL_SERVER
    assert server.subtype == "ollama-server"
    assert worker.category is Category.MODEL_SERVER
    assert worker.subtype == "llama-server (inference)"


def test_llama_server_inference_process_is_high_risk():
    r = classify("/usr/local/bin/llama-server", ["llama-server", "--model", "x"], "llama-server")
    assert r.category is Category.MODEL_SERVER
    assert r.risk is Risk.HIGH


def test_claude_desktop_main_is_high_risk():
    r = classify("/Applications/Claude.app/Contents/MacOS/Claude", ["Claude"], "Claude")
    assert r.category is Category.CLAUDE_DESKTOP
    assert r.subtype == "app"
    assert r.risk is Risk.HIGH


def test_claude_helper_is_low_risk():
    r = classify(
        "/Applications/Claude.app/Contents/Frameworks/Claude Helper.app/Contents/MacOS/Claude Helper",
        ["Claude Helper", "--type=utility"],
        "Claude Helper",
    )
    assert r.category is Category.CLAUDE_DESKTOP
    assert r.subtype == "helper"
    assert r.risk is Risk.LOW


def test_chatgpt_codex_service():
    r = classify(
        "/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Versions/1/Helpers/Codex (Service).app/Contents/MacOS/Codex (Service)",
        ["Codex (Service)"],
        "Codex (Service)",
    )
    assert r.category is Category.CHATGPT_CODEX
    assert r.subtype == "codex-service"
    assert r.risk is Risk.MEDIUM


def test_chatgpt_codex_renderer_is_low_risk():
    r = classify(
        "/Applications/ChatGPT.app/Contents/Frameworks/Codex Framework.framework/Versions/1/Helpers/Codex (Renderer).app/Contents/MacOS/Codex (Renderer)",
        ["Codex (Renderer)"],
        "Codex (Renderer)",
    )
    assert r.risk is Risk.LOW


def test_cursor_main_app():
    r = classify("/Applications/Cursor.app/Contents/MacOS/Cursor", ["Cursor"], "Cursor")
    assert r.category is Category.CURSOR
    assert r.risk is Risk.HIGH


def test_copilot_cli_session():
    r = classify(
        "/Users/testuser/Library/Caches/github-copilot-sdk/cli/1.0.80/copilot",
        ["copilot", "--server", "--stdio"],
        "copilot",
    )
    assert r.category is Category.COPILOT_SESSION
    assert r.risk is Risk.HIGH


def test_windows_copilot_cli_session():
    r = classify(
        r"C:\Users\user\AppData\Local\github-copilot-sdk\cli\1.0.0\copilot.exe",
        ["copilot.exe", "--server", "--stdio"],
        "copilot.exe",
    )
    assert r.category is Category.COPILOT_SESSION
    assert r.risk is Risk.HIGH


def test_copilot_vscode_extension_headless():
    r = classify(
        "/Applications/Visual Studio Code.app/Contents/Frameworks/Code Helper.app/Contents/MacOS/Code Helper",
        [
            "Code Helper",
            "/Applications/Visual Studio Code.app/Contents/Resources/app/node_modules.asar.unpacked/@github/copilot-darwin-arm64/index.js",
            "--headless",
            "--stdio",
        ],
        "Code Helper",
    )
    assert r.category is Category.COPILOT_EXTENSION
    assert r.risk is Risk.HIGH


def test_agency_mcp_simple_tool():
    r = classify("/Users/testuser/.local/bin/agency", ["agency", "mcp", "icm"], "agency")
    assert r.category is Category.MCP_TOOL
    assert r.subtype == "icm"
    assert r.risk is Risk.LOW


def test_windows_agency_mcp_path_is_classified():
    r = classify(
        r"C:\Users\user\.local\bin\agency.exe",
        ["agency.exe", "mcp", "icm"],
        "agency.exe",
    )
    assert r.category is Category.MCP_TOOL
    assert r.subtype == "icm"


def test_agency_mcp_native_tool():
    r = classify("/Users/testuser/.local/bin/agency", ["agency", "mcp", "native", "kusto"], "agency")
    assert r.category is Category.MCP_TOOL
    assert r.subtype == "kusto (native)"


def test_agency_mcp_tool_with_flags():
    r = classify(
        "/Users/testuser/.local/bin/agency",
        ["agency", "mcp", "ado", "--organization", "1esgitops"],
        "agency",
    )
    assert r.subtype == "ado"


def test_workiq_mcp_direct():
    r = classify("/opt/homebrew/lib/node_modules/@microsoft/workiq/bin/osx-arm64/workiq", ["workiq", "mcp"], "workiq")
    assert r.category is Category.WORKIQ
    assert r.risk is Risk.LOW


def test_workiq_mcp_via_npm_exec():
    r = classify("/usr/local/bin/npm", ["npm", "exec", "@microsoft/workiq", "mcp"], "npm")
    assert r.category is Category.WORKIQ


def test_computer_use_mcp_plugin():
    r = classify(
        "/Users/testuser/Library/Caches/copilot/pkg/darwin-arm64/1.0.80/plugins/computer-use/computer-use-mcp",
        ["computer-use-mcp"],
        "computer-use-mcp",
    )
    assert r.category is Category.MCP_TOOL
    assert r.subtype == "computer-use"


def test_generic_unrecognized_mcp_is_surfaced_not_dropped():
    r = classify("/usr/local/bin/some-tool", ["some-tool", "mcp"], "some-tool")
    assert r.category is Category.OTHER_AGENT
    assert r.subtype == "unknown-mcp"


def test_unrelated_system_agents_are_excluded():
    """The word 'agent' alone must never be enough to match."""
    system_agents = [
        ("/usr/libexec/trustd", ["trustd", "--agent"], "trustd"),
        ("/usr/libexec/ContinuityCaptureAgent", ["ContinuityCaptureAgent", "server"], "ContinuityCaptureAgent"),
        ("/usr/libexec/gamecontrolleragentd", ["gamecontrolleragentd"], "gamecontrolleragentd"),
        ("/usr/libexec/assessmentagent", ["assessmentagent"], "assessmentagent"),
        ("/usr/libexec/PowerUIAgent", ["PowerUIAgent"], "PowerUIAgent"),
        ("/usr/sbin/distnoted", ["distnoted", "agent"], "distnoted"),
        ("/System/Library/CoreServices/iconservicesagent", ["iconservicesagent"], "iconservicesagent"),
    ]
    for exe, cmdline, name in system_agents:
        assert classify(exe, cmdline, name) is None, f"{name} should not be classified as an agent"


def test_unrelated_apps_are_excluded():
    r = classify("/System/Applications/Calculator.app/Contents/MacOS/Calculator", ["Calculator"], "Calculator")
    assert r is None


def test_handles_missing_exe_and_cmdline_gracefully():
    assert classify(None, None, None) is None
    assert classify("", [], "") is None
