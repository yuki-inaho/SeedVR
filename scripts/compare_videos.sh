#!/usr/bin/env bash
# Build a side-by-side comparison video (left | right) with a top-left text
# label on each side. Both inputs are scaled to a common height before hstack.
#
# Labels and the output name are generic and user-controlled — the script does
# NOT embed the source directory or dataset name. Defaults are "Raw"/"Restored".
#
# Usage:
#   scripts/compare_videos.sh <left.mp4> <right.mp4> <output.mp4> [left_label] [right_label] [height]
# Example:
#   scripts/compare_videos.sh raw/clip.mp4 restored/clip.mp4 comparison.mp4 "Raw" "Restored" 1024
set -euo pipefail

LEFT="${1:?usage: compare_videos.sh <left.mp4> <right.mp4> <output.mp4> [left_label] [right_label] [height]}"
RIGHT="${2:?right video required}"
OUT="${3:?output mp4 path required}"
LLABEL="${4:-Raw}"
RLABEL="${5:-Restored}"
H="${6:-720}"

# Locate a usable font for drawtext.
FONT=""
for f in \
  /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
  /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf \
  /usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf \
  /usr/share/fonts/truetype/freefont/FreeSansBold.ttf; do
  [ -f "$f" ] && FONT="$f" && break
done
[ -n "$FONT" ] || { echo "error: no TrueType font found for drawtext" >&2; exit 1; }

FS=$(( H / 22 ))   # label font size relative to height
mkdir -p "$(dirname "$OUT")"

ffmpeg -y -i "$LEFT" -i "$RIGHT" -filter_complex "\
[0:v]scale=-2:${H},setsar=1,\
drawtext=fontfile='${FONT}':text='${LLABEL}':x=20:y=20:fontsize=${FS}:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=10[l];\
[1:v]scale=-2:${H},setsar=1,\
drawtext=fontfile='${FONT}':text='${RLABEL}':x=20:y=20:fontsize=${FS}:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=10[r];\
[l][r]hstack=inputs=2[v]" \
  -map "[v]" -c:v libx264 -pix_fmt yuv420p -crf 16 -movflags +faststart "$OUT"
echo "wrote $OUT"
