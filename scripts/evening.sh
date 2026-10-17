#!/bin/sh
# Record one evening from the real app (PLAN.md phase 8).
#
# `--record` renders one frame per time-lapse step into a directory; this script
# encodes those frames into the animated WebP and the still that the README
# shows, so both come from the binary rather than from a separate renderer.
#
# On a hybrid-graphics laptop the discrete GPU must be chosen (`prime-run` on
# this machine); override with `GPU=` if the executable is elsewhere, e.g.
#   GPU= ./scripts/evening.sh
set -eu

GPU="${GPU-prime-run}"
BIRDS="${BIRDS:-8000}"
SEED="${SEED:-1}"
MINUTES="${MINUTES:-1}"
FRAMES="${FRAMES:-180}"
FPS="${FPS:-20}"
OUT="${OUT:-docs/evening.webp}"
STILL="${STILL:-docs/evening.png}"
DIR="${DIR:-$(mktemp -d)}"

cargo build -p murmuration-gpu --release
BIN=target/release/murmuration-gpu

mkdir -p "$DIR"
# Never erase an existing recording or an unrelated directory supplied by DIR.
if [ -n "$(find "$DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]; then
    echo "recording directory is not empty: $DIR" >&2
    exit 1
fi
# shellcheck disable=SC2086
$GPU "$BIN" --record "$DIR" --birds "$BIRDS" --seed "$SEED" \
    --arc-minutes "$MINUTES" --frames "$FRAMES"

# The frames are of varying brightness (a gold arrival and a dark night), so a
# lossy encode with a decent quality is smaller than a lossless one and does not
# band.
ffmpeg -y -loglevel error -framerate "$FPS" -i "$DIR/frame-%04d.png" \
    -vf "scale=720:-1" -c:v libwebp_anim -lossless 0 -q:v 72 -loop 0 "$OUT"
# A representative still: the display flock, about halfway through the evening.
cp "$DIR/frame-$(printf '%04d' $((FRAMES / 2))).png" "$STILL"

echo "wrote $OUT and $STILL from $DIR"
