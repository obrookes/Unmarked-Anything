#!/usr/bin/env bash
set -euo pipefail

# Batch-reduce video frame rate using ffmpeg while preserving originals.

usage() {
  cat <<'USAGE'
Usage:
  ./reduce_video_fps.sh [directory] [-f fps] [--crf N] [--preset NAME]

Options:
  -f, --fps N       Output frames per second (default: 5).
  --crf N           x264 CRF value (default: 18).
  --preset NAME     x264 preset (default: medium).
  -h, --help        Show this help.

Behavior:
  Creates duplicate files with '-<fps>fps' added before the extension.
  Example: input.MP4 -> input-5fps.MP4

Examples:
  ./reduce_video_fps.sh assets/videos
  ./reduce_video_fps.sh assets/videos --fps 5
  ./reduce_video_fps.sh . --fps 10 --crf 20 --preset slow
USAGE
}

DIR="."
FPS="5"
CRF="18"
PRESET="medium"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--fps)
      [[ $# -lt 2 ]] && { echo "Error: missing value for $1" >&2; exit 1; }
      FPS="$2"
      shift 2
      ;;
    --crf)
      [[ $# -lt 2 ]] && { echo "Error: missing value for $1" >&2; exit 1; }
      CRF="$2"
      shift 2
      ;;
    --preset)
      [[ $# -lt 2 ]] && { echo "Error: missing value for $1" >&2; exit 1; }
      PRESET="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Error: unknown option '$1'" >&2
      usage
      exit 1
      ;;
    *)
      DIR="$1"
      shift
      ;;
  esac
done

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$DIR" ]]; then
  echo "Error: directory not found: $DIR" >&2
  exit 1
fi

if ! [[ "$FPS" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: fps must be numeric (e.g. 5 or 7.5)" >&2
  exit 1
fi
if ! awk -v fps="$FPS" 'BEGIN{ exit !(fps > 0) }'; then
  echo "Error: fps must be greater than 0" >&2
  exit 1
fi

if ! [[ "$CRF" =~ ^[0-9]+$ ]]; then
  echo "Error: crf must be an integer" >&2
  exit 1
fi

suffix_fps="$FPS"

shopt -s nullglob nocaseglob
files=(
  "$DIR"/*.mp4 "$DIR"/*.mov "$DIR"/*.mkv "$DIR"/*.avi "$DIR"/*.m4v
  "$DIR"/*.MP4 "$DIR"/*.MOV "$DIR"/*.MKV "$DIR"/*.AVI "$DIR"/*.M4V
)
shopt -u nocaseglob

if [[ ${#files[@]} -eq 0 ]]; then
  echo "No video files found in: $DIR"
  exit 0
fi

echo "Directory: $DIR"
echo "Target FPS: $FPS"
echo "CRF: $CRF"
echo "Preset: $PRESET"

for input in "${files[@]}"; do
  [[ -f "$input" ]] || continue

  base="${input%.*}"
  ext="${input##*.}"
  output="${base}-${suffix_fps}fps.${ext}"

  if [[ "$input" == "$output" ]]; then
    echo "Skipping already-reduced file: $input"
    continue
  fi

  echo "Reducing FPS -> $output"
  ffmpeg -y -i "$input" -vf "fps=${FPS}" -c:v libx264 -crf "$CRF" -preset "$PRESET" -c:a copy "$output"
done

echo "Done."
