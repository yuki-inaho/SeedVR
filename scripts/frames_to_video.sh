#!/usr/bin/env bash
# Encode a folder of sequentially-named frames into an mp4.
#
# Output naming is whatever you pass as <output.mp4> — the script does NOT embed
# the source directory or dataset name anywhere. Use a generic name (e.g.
# output.mp4) if you want to avoid leaking source identifiers.
#
# Usage:
#   scripts/frames_to_video.sh <frames_dir> <output.mp4> [fps] [glob]
# Example:
#   scripts/frames_to_video.sh ./out output.mp4 8 '*.png'
set -euo pipefail

IN="${1:?usage: frames_to_video.sh <frames_dir> <output.mp4> [fps] [glob]}"
OUT="${2:?output mp4 path required}"
FPS="${3:-8}"
GLOB="${4:-*.png}"

mkdir -p "$(dirname "$OUT")"
# -pattern_type glob sorts lexicographically; zero-padded sequential names work.
ffmpeg -y -framerate "$FPS" -pattern_type glob -i "$IN/$GLOB" \
  -c:v libx264 -pix_fmt yuv420p -crf 16 -movflags +faststart "$OUT"
echo "wrote $OUT"
