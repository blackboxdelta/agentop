from __future__ import annotations

from pathlib import Path
import struct

from scripts.verify_binary import detect_architecture


def test_detects_macos_arm64_macho(tmp_path):
    binary = tmp_path / "agentop"
    binary.write_bytes(b"\xcf\xfa\xed\xfe" + struct.pack("<I", 0x0100000C))
    assert detect_architecture(binary) == "arm64"


def test_detects_windows_x64_pe(tmp_path):
    data = bytearray(128)
    data[:2] = b"MZ"
    struct.pack_into("<I", data, 0x3C, 64)
    data[64:68] = b"PE\0\0"
    struct.pack_into("<H", data, 68, 0x8664)
    binary = tmp_path / "agentop.exe"
    binary.write_bytes(data)
    assert detect_architecture(binary) == "x86_64"
