#!/bin/sh
set -eu

REPO="blackboxdelta/agentop"
INSTALL_DIR="${AGENTOP_INSTALL_DIR:-$HOME/.local/bin}"

if [ "$(uname -s)" != "Darwin" ] || [ "$(uname -m)" != "arm64" ]; then
  echo "agentop: this installer supports Apple Silicon macOS only." >&2
  exit 1
fi

if [ -n "${AGENTOP_VERSION:-}" ]; then
  TAG="$AGENTOP_VERSION"
else
  TAG="$(
    curl -fsSL "https://api.github.com/repos/$REPO/releases/latest" |
      sed -n 's/.*"tag_name":[[:space:]]*"\([^"]*\)".*/\1/p' |
      head -n 1
  )"
fi

if [ -z "$TAG" ]; then
  echo "agentop: could not determine the latest release." >&2
  exit 1
fi

BASE_URL="https://github.com/$REPO/releases/download/$TAG"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT INT TERM

curl -fsSL "$BASE_URL/agentop-macos-arm64.tar.gz" \
  -o "$TMP_DIR/agentop.tar.gz"
curl -fsSL "$BASE_URL/agentop-macos-arm64.sha256" \
  -o "$TMP_DIR/agentop.sha256"

EXPECTED="$(awk '{print $1}' "$TMP_DIR/agentop.sha256")"
ACTUAL="$(shasum -a 256 "$TMP_DIR/agentop.tar.gz" | awk '{print $1}')"
if [ "$EXPECTED" != "$ACTUAL" ]; then
  echo "agentop: checksum verification failed." >&2
  exit 1
fi

mkdir -p "$TMP_DIR/package" "$INSTALL_DIR"
tar -xzf "$TMP_DIR/agentop.tar.gz" -C "$TMP_DIR/package"
install -m 755 "$TMP_DIR/package/agentop" "$INSTALL_DIR/agentop"

echo "agentop $TAG installed to $INSTALL_DIR/agentop"
case ":$PATH:" in
  *":$INSTALL_DIR:"*) ;;
  *) echo "Add $INSTALL_DIR to PATH, then run: agentop" ;;
esac
