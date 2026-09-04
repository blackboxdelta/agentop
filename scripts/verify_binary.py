"""Verify release binary architecture and startup behavior."""
from __future__ import annotations

import argparse
from pathlib import Path
import struct
import subprocess


def detect_architecture(path: Path) -> str:
    data = path.read_bytes()
    if data[:2] == b"MZ":
        pe_offset = struct.unpack_from("<I", data, 0x3C)[0]
        if data[pe_offset : pe_offset + 4] != b"PE\0\0":
            raise ValueError("invalid PE signature")
        machine = struct.unpack_from("<H", data, pe_offset + 4)[0]
        if machine == 0x8664:
            return "x86_64"
        if machine == 0xAA64:
            return "arm64"
        return f"pe-machine-{machine:#x}"
    magic = data[:4]
    if magic == b"\xcf\xfa\xed\xfe":
        cpu_type = struct.unpack_from("<I", data, 4)[0]
        if cpu_type == 0x0100000C:
            return "arm64"
        if cpu_type == 0x01000007:
            return "x86_64"
        return f"macho-cpu-{cpu_type:#x}"
    raise ValueError("unsupported executable format")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    parser.add_argument("expected_arch", choices=("arm64", "x86_64"))
    args = parser.parse_args()

    detected = detect_architecture(args.binary)
    if detected != args.expected_arch:
        raise SystemExit(
            f"architecture mismatch: expected {args.expected_arch}, got {detected}"
        )
    subprocess.run([args.binary, "--version"], check=True)
    subprocess.run([args.binary, "--help"], check=True, stdout=subprocess.DEVNULL)
    print(f"verified {args.binary}: {detected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
