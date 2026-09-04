# Distribution and installation

`agentop` publishes two standalone artifacts from tagged releases:

| Artifact | Target |
|---|---|
| `agentop-macos-arm64.tar.gz` | Apple Silicon macOS |
| `agentop-windows-x64.zip` | 64-bit Windows |

The GitHub workflow runs the complete test suite, builds from the checked-in
`agentop.spec`, verifies the PE/Mach-O architecture, launches the frozen
binary with `--version` and `--help`, then packages it with this documentation,
the example configuration, license, and a SHA-256 checksum.

## Recommended one-line install

macOS Apple Silicon:

```bash
curl -fsSL https://raw.githubusercontent.com/blackboxdelta/agentop/main/install.sh | sh
```

Windows x64 PowerShell:

```powershell
irm https://raw.githubusercontent.com/blackboxdelta/agentop/main/install.ps1 | iex
```

## macOS Apple Silicon

```bash
shasum -a 256 -c agentop-macos-arm64.sha256
tar -xzf agentop-macos-arm64.tar.gz
chmod +x agentop
sudo mv agentop /usr/local/bin/
agentop --version
```

Release binaries are not notarized unless the project configures Apple signing
secrets. For an unsigned community build, macOS may require explicit approval
in **System Settings > Privacy & Security** after the first launch. Do not
disable Gatekeeper system-wide.

## Windows x64

```powershell
$expected = (Get-Content .\agentop-windows-x64.sha256).Split()[0]
$actual = (Get-FileHash .\agentop-windows-x64.zip -Algorithm SHA256).Hash.ToLower()
if ($actual -ne $expected) { throw "Checksum mismatch" }

Expand-Archive .\agentop-windows-x64.zip -DestinationPath .\agentop
.\agentop\agentop.exe --version
```

Move the extracted directory somewhere permanent and add it to the user
`PATH`. Windows SmartScreen may ask for confirmation when the project has not
configured a code-signing certificate.

## Source installation

Python 3.11 or newer and `uv` are required:

```bash
git clone https://github.com/blackboxdelta/agentop.git
cd agentop
uv tool install .
agentop --help
```

## Release maintainers

1. Update `src/agentop/__init__.py`, `pyproject.toml`, and the root package
   version in `uv.lock`.
2. Run:

   ```bash
   uv sync --group dev --locked
   uv run pytest -q
   uv run pyinstaller --clean agentop.spec
   uv run python scripts/verify_binary.py dist/agentop arm64
   ```

3. Push a tag such as `v0.3.0`.
4. Confirm both GitHub Actions matrix jobs pass and that their checksums match
   the release attachments.
