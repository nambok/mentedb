#!/usr/bin/env bash
# Sync SDK versions with the workspace Cargo.toml version.
# Run this after bumping the workspace version to keep npm/PyPI in sync.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION=$(cargo metadata --manifest-path "$REPO_ROOT/Cargo.toml" --no-deps --format-version 1 \
  | jq -r '.packages[] | select(.name == "mentedb") | .version')

if [ -z "$VERSION" ]; then
  echo "❌ Could not determine workspace version"
  exit 1
fi

echo "📦 Syncing SDK versions to $VERSION"

# TypeScript: package.json
TS_PKG="$REPO_ROOT/sdks/typescript/package.json"
if [ -f "$TS_PKG" ]; then
  jq --arg v "$VERSION" '.version = $v' "$TS_PKG" > "$TS_PKG.tmp" && mv "$TS_PKG.tmp" "$TS_PKG"
  echo "  ✅ sdks/typescript/package.json → $VERSION"
fi

# TypeScript: Cargo.toml
TS_CARGO="$REPO_ROOT/sdks/typescript/Cargo.toml"
if [ -f "$TS_CARGO" ]; then
  sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$TS_CARGO" && rm -f "$TS_CARGO.bak"
  echo "  ✅ sdks/typescript/Cargo.toml → $VERSION"
fi

# Python: pyproject.toml
PY_PROJ="$REPO_ROOT/sdks/python/pyproject.toml"
if [ -f "$PY_PROJ" ]; then
  sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$PY_PROJ" && rm -f "$PY_PROJ.bak"
  echo "  ✅ sdks/python/pyproject.toml → $VERSION"
fi

# Python: Cargo.toml
PY_CARGO="$REPO_ROOT/sdks/python/Cargo.toml"
if [ -f "$PY_CARGO" ]; then
  sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$PY_CARGO" && rm -f "$PY_CARGO.bak"
  echo "  ✅ sdks/python/Cargo.toml → $VERSION"
fi

echo "🎉 All SDK versions synced to $VERSION"
