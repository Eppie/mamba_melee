#!/usr/bin/env bash
# Download + extract the Slippi Playback Dolphin AppImage.
# After this finishes, point the regen driver at the squashfs-root via:
#   export SLIPPI_PLAYBACK_DIR=$PWD/.playback/squashfs-root
set -euo pipefail

DEST="${1:-$PWD/.playback}"
VERSION="${SLIPPI_PLAYBACK_VERSION:-3.5.2}"
URL="https://github.com/project-slippi/Ishiiruka-Playback/releases/download/v${VERSION}/playback-${VERSION}-Linux.zip"

mkdir -p "$DEST"
cd "$DEST"

if [ ! -f "Slippi_Playback-x86_64.AppImage" ]; then
    echo "Downloading $URL ..."
    curl -sSL -o playback.zip "$URL"
    unzip -q -o playback.zip
    rm -f playback.zip
fi

if [ ! -d "squashfs-root" ]; then
    echo "Extracting AppImage ..."
    chmod +x Slippi_Playback-x86_64.AppImage
    ./Slippi_Playback-x86_64.AppImage --appimage-extract > /dev/null
fi

if [ ! -f squashfs-root/AppRun ]; then
    echo "ERROR: extraction failed — squashfs-root/AppRun not found" >&2
    exit 1
fi

echo
echo "Done. Set this in your shell:"
echo "  export SLIPPI_PLAYBACK_DIR=$DEST/squashfs-root"
