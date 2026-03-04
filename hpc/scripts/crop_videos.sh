#!/usr/bin/env bash
set -euo pipefail

# Batch-crop bottom percentage from videos using ffmpeg.
# Default crop percentage: 9.75

usage() {
  cat <<'USAGE'
Usage:
  ./crop_videos.sh (--overwrite | --suffix) [directory] [-p percent]

Options:
  --overwrite       Replace original files with cropped versions.
  --suffix          Create new files with '-crop' suffix before extension.
  -p, --percent N   Bottom percent to remove (default: 9.75).
  -h, --help        Show this help.

Examples:
  ./crop_videos.sh --suffix /path/to/videos
  ./crop_videos.sh --overwrite . -p 9.75
USAGE
}

MODE=""
DIR="."
PERCENT="9.75"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --overwrite)
      [[ -n "$MODE" ]] && { echo "Error: choose only one of --overwrite or --suffix" >&2; exit 1; }
      MODE="overwrite"
      shift
      ;;
    --suffix)
      [[ -n "$MODE" ]] && { echo "Error: choose only one of --overwrite or --suffix" >&2; exit 1; }
      MODE="suffix"
      shift
      ;;
    -p|--percent)
      [[ $# -lt 2 ]] && { echo "Error: missing value for $1" >&2; exit 1; }
      PERCENT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -* )
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

if [[ -z "$MODE" ]]; then
  echo "Error: you must choose one mode: --overwrite or --suffix" >&2
  usage
  exit 1
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg not found in PATH" >&2
  exit 1
fi

if [[ ! -d "$DIR" ]]; then
  echo "Error: directory not found: $DIR" >&2
  exit 1
fi

# Validate percent is numeric and between 0 and <100.
if ! [[ "$PERCENT" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "Error: percent must be numeric (e.g. 9.75)" >&2
  exit 1
fi
if ! awk -v p="$PERCENT" 'BEGIN{ exit !(p >= 0 && p < 100) }'; then
  echo "Error: percent must be between 0 and <100" >&2
  exit 1
fi

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

echo "Mode: $MODE"
echo "Directory: $DIR"
echo "Bottom crop percent: $PERCENT"

for input in "${files[@]}"; do
  [[ -f "$input" ]] || continue

  base="${input%.*}"
  ext="${input##*.}"

  if [[ "$MODE" == "suffix" ]]; then
    output="${base}-crop.${ext}"
    echo "Cropping -> $output"
    ffmpeg -y -i "$input" -vf "crop=iw:ih*(1-${PERCENT}/100):0:0" -c:a copy "$output"
  else
    tmp_output="${base}.crop-tmp.${ext}"
    echo "Cropping (overwrite) -> $input"
    ffmpeg -y -i "$input" -vf "crop=iw:ih*(1-${PERCENT}/100):0:0" -c:a copy "$tmp_output"
    mv -f "$tmp_output" "$input"
  fi
done

echo "Done."
