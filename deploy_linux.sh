#!/usr/bin/env bash
# deploy_linux.sh — build the Rust renderer and deploy it to the Blender addon on Linux.
# Linux counterpart of deploy_blender.ps1. The Rust crate builds a cdylib (libignis_rt.so) that the
# addon loads via ctypes; display goes through the readback fallback (no Win32 GL interop needed).
#
#   ./deploy_linux.sh            # release build + copy .so into blender/ignis_rt/lib/
#   ./deploy_linux.sh --debug    # faster-to-compile debug build (slow at runtime)
#   ./deploy_linux.sh --symlink  # also symlink the addon into Blender's extensions dir (live Python edits)
#   ./deploy_linux.sh --no-build # skip cargo, just (re)deploy the last-built .so
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROFILE="release"; FLAG="--release"; SYMLINK=0; BUILD=1
for arg in "$@"; do
  case "$arg" in
    --debug)    PROFILE="debug"; FLAG="" ;;
    --symlink)  SYMLINK=1 ;;
    --no-build) BUILD=0 ;;
    *) echo "unknown arg: $arg" >&2; exit 1 ;;
  esac
done

if [[ "$BUILD" == "1" ]]; then
  echo "==> cargo build ($PROFILE)"
  ( cd "$REPO/rust" && cargo build $FLAG )
fi

SO="$REPO/rust/target/$PROFILE/libignis_rt.so"
DEST="$REPO/blender/ignis_rt/lib"
[[ -f "$SO" ]] || { echo "!! not found: $SO (build first)" >&2; exit 1; }
mkdir -p "$DEST"
cp -f "$SO" "$DEST/"
echo "==> deployed $(basename "$SO") -> blender/ignis_rt/lib/"

# DLSS snippets (Linux): copy them next to the .so so NGX can dlopen them at runtime. Needs
# IGNIS_NGX_ROOT pointing at the DLSS SDK (same var build.rs uses to enable have_ngx). Variant:
# dev = no app-id allowlist required (adds a small on-screen watermark); rel = production (allowlisted).
NGX_VARIANT="${IGNIS_NGX_VARIANT:-dev}"
if [[ -n "${IGNIS_NGX_ROOT:-}" && -d "$IGNIS_NGX_ROOT/lib/Linux_x86_64/$NGX_VARIANT" ]]; then
  for so in "$IGNIS_NGX_ROOT"/lib/Linux_x86_64/"$NGX_VARIANT"/libnvidia-ngx-dlss*.so.*; do
    [[ -e "$so" ]] || continue
    base="$(basename "$so")"
    cp -f "$so" "$DEST/"
    ln -sf "$base" "$DEST/${base%.so.*}.so"   # unversioned symlink alongside the versioned snippet
  done
  echo "==> copied DLSS snippets ($NGX_VARIANT) from IGNIS_NGX_ROOT"
fi

if [[ "$SYMLINK" == "1" ]]; then
  # Symlink the addon into Blender's local extensions repo so Python edits are picked up without reinstalling.
  BL_EXT="$(ls -d "$HOME"/.config/blender/*/extensions/user_default 2>/dev/null | sort -V | tail -1 || true)"
  if [[ -z "$BL_EXT" ]]; then
    echo "!! Blender extensions dir not found under ~/.config/blender/*/extensions/user_default"
    echo "   Install the addon once in Blender (Install from Disk), then re-run with --symlink."
  else
    LINK="$BL_EXT/ignis_rt"
    # Drop a prior Install-from-Disk copy (a real dir) so the symlink takes instead of nesting inside it.
    [[ -e "$LINK" && ! -L "$LINK" ]] && rm -rf "$LINK"
    ln -sfn "$REPO/blender/ignis_rt" "$LINK"
    echo "==> symlinked addon -> $LINK  (restart Blender)"
  fi
fi

echo "Done. In Blender: enable 'Ignis RT' -> set it as the render engine -> viewport shading: Rendered."
