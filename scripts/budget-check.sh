#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
usage: budget-check.sh [OPTIONS] [CSV_PATH]

OPTIONS:
  --seeds N            Number of seeds (default: 5)
  --epochs M           Number of epochs (default: 5)
  --wall-hours-green G Green threshold in hours (default: 20)
  --wall-hours-red R   Red threshold in hours (default: 24)
  --peak-mem-gb P      Peak memory threshold in GB (default: 40)
  --help               Print this usage and exit 0
USAGE
}

die_usage() {
  echo "ERROR: $1" >&2
  usage >&2
  exit 2
}

require_positive_int() {
  local name="$1"
  local value="$2"

  if ! awk -v value="$value" 'BEGIN { exit (value ~ /^[0-9][0-9]*$/ && value > 0 ? 0 : 1) }'; then
    echo "ERROR: $name must be a positive integer" >&2
    exit 2
  fi
}

require_positive_number() {
  local name="$1"
  local value="$2"

  if ! awk -v value="$value" 'BEGIN { exit (value ~ /^[0-9][0-9]*([.][0-9][0-9]*)?$/ && value > 0 ? 0 : 1) }'; then
    echo "ERROR: $name must be a positive number" >&2
    exit 2
  fi
}

format_one_decimal() {
  local value="$1"
  awk -v value="$value" 'BEGIN { printf "%.1f", value + 0 }'
}

format_compact_number() {
  local value="$1"
  awk -v value="$value" 'BEGIN { printf "%g", value + 0 }'
}

SEEDS=5
EPOCHS=5
WALL_HOURS_GREEN=20
WALL_HOURS_RED=24
PEAK_MEM_GB=40
CSV_PATH="/tmp/voi-210-timing-probe.csv"
CSV_PATH_SET=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --seeds)
      [ "$#" -ge 2 ] || die_usage "--seeds requires a value"
      SEEDS="$2"
      shift 2
      ;;
    --epochs)
      [ "$#" -ge 2 ] || die_usage "--epochs requires a value"
      EPOCHS="$2"
      shift 2
      ;;
    --wall-hours-green)
      [ "$#" -ge 2 ] || die_usage "--wall-hours-green requires a value"
      WALL_HOURS_GREEN="$2"
      shift 2
      ;;
    --wall-hours-red)
      [ "$#" -ge 2 ] || die_usage "--wall-hours-red requires a value"
      WALL_HOURS_RED="$2"
      shift 2
      ;;
    --peak-mem-gb)
      [ "$#" -ge 2 ] || die_usage "--peak-mem-gb requires a value"
      PEAK_MEM_GB="$2"
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    --)
      shift
      [ "$#" -le 1 ] || die_usage "too many positional arguments"
      if [ "$#" -eq 1 ]; then
        [ "$CSV_PATH_SET" -eq 0 ] || die_usage "too many CSV paths"
        CSV_PATH="$1"
        CSV_PATH_SET=1
        shift
      fi
      ;;
    -*)
      die_usage "unknown option: $1"
      ;;
    *)
      [ "$CSV_PATH_SET" -eq 0 ] || die_usage "too many CSV paths"
      CSV_PATH="$1"
      CSV_PATH_SET=1
      shift
      ;;
  esac
done

require_positive_int "--seeds" "$SEEDS"
require_positive_int "--epochs" "$EPOCHS"
require_positive_number "--wall-hours-green" "$WALL_HOURS_GREEN"
require_positive_number "--wall-hours-red" "$WALL_HOURS_RED"
require_positive_number "--peak-mem-gb" "$PEAK_MEM_GB"

if ! awk -v green="$WALL_HOURS_GREEN" -v red_h="$WALL_HOURS_RED" 'BEGIN { exit (red_h > green ? 0 : 1) }'; then
  echo "ERROR: --wall-hours-red must be greater than --wall-hours-green" >&2
  exit 2
fi

if [ ! -r "$CSV_PATH" ]; then
  echo "ERROR: CSV file is not readable: $CSV_PATH" >&2
  exit 1
fi

EXPECTED_HEADER="condition,seed,final_val_f1,wall_time_s,peak_memory_bytes"
if ! IFS= read -r HEADER < "$CSV_PATH"; then
  echo "ERROR: CSV is empty: $CSV_PATH" >&2
  exit 1
fi
HEADER="${HEADER%$'\r'}"

if [ "$HEADER" != "$EXPECTED_HEADER" ]; then
  echo "ERROR: CSV header mismatch" >&2
  exit 1
fi

if ! AWK_OUT=$(awk '
BEGIN {
  FS = ","
}

function fail(message) {
  print message > "/dev/stderr"
  had_error = 1
  exit 1
}

function is_number(value) {
  return value ~ /^[0-9][0-9]*(\.[0-9][0-9]*)?([eE][+-]?[0-9][0-9]*)?$/
}

NR == 1 {
  next
}

NF == 0 {
  next
}

{
  gsub(/\r/, "", $0)

  if (NF != 5) {
    fail("ERROR: CSV row " NR " must have 5 fields")
  }

  condition = $1
  wall = $4
  mem = $5

  if (!is_number(wall)) {
    fail("ERROR: CSV row " NR " has non-numeric wall_time_s")
  }
  if (!is_number(mem)) {
    fail("ERROR: CSV row " NR " has non-numeric peak_memory_bytes")
  }

  row_count += 1
  wall_num = wall + 0
  mem_num = mem + 0
  sum_wall += wall_num

  if (mem_num > max_mem) {
    max_mem = mem_num
  }

  cond_wall[condition] += wall_num
  if (mem_num > cond_mem[condition]) {
    cond_mem[condition] = mem_num
  }
}

END {
  if (had_error) {
    exit 1
  }
  if (row_count < 1) {
    print "ERROR: CSV has no data rows" > "/dev/stderr"
    exit 1
  }

  printf "%d\n", row_count
  printf "%.6f\n", sum_wall
  printf "%.6f\n", max_mem
  printf "%.6f\n", cond_wall["sensors-only"] + 0
  printf "%.6f\n", cond_mem["sensors-only"] + 0
  printf "%.6f\n", cond_wall["vision+text"] + 0
  printf "%.6f\n", cond_mem["vision+text"] + 0
  printf "%.6f\n", cond_wall["all-three"] + 0
  printf "%.6f\n", cond_mem["all-three"] + 0
}
' "$CSV_PATH"); then
  exit 1
fi

ROW_COUNT=$(awk 'NR == 1 { print; exit }' <<< "$AWK_OUT")
SUM_WALL_S=$(awk 'NR == 2 { print; exit }' <<< "$AWK_OUT")
MAX_MEM_BYTES=$(awk 'NR == 3 { print; exit }' <<< "$AWK_OUT")
SENSORS_WALL_S=$(awk 'NR == 4 { print; exit }' <<< "$AWK_OUT")
SENSORS_MEM_BYTES=$(awk 'NR == 5 { print; exit }' <<< "$AWK_OUT")
VISION_TEXT_WALL_S=$(awk 'NR == 6 { print; exit }' <<< "$AWK_OUT")
VISION_TEXT_MEM_BYTES=$(awk 'NR == 7 { print; exit }' <<< "$AWK_OUT")
ALL_THREE_WALL_S=$(awk 'NR == 8 { print; exit }' <<< "$AWK_OUT")
ALL_THREE_MEM_BYTES=$(awk 'NR == 9 { print; exit }' <<< "$AWK_OUT")

PROJECTED_WALL_H=$(awk \
  -v sum_wall_s="$SUM_WALL_S" \
  -v seeds="$SEEDS" \
  -v epochs="$EPOCHS" \
  'BEGIN { printf "%.6f", (sum_wall_s * seeds * epochs) / 3600.0 }')
PROJECTED_PEAK_GB=$(awk \
  -v peak_bytes="$MAX_MEM_BYTES" \
  'BEGIN { printf "%.6f", peak_bytes / (1024.0 * 1024.0 * 1024.0) }')

SENSORS_MEM_GB=$(awk -v bytes="$SENSORS_MEM_BYTES" 'BEGIN { printf "%.6f", bytes / (1024.0 * 1024.0 * 1024.0) }')
VISION_TEXT_MEM_GB=$(awk -v bytes="$VISION_TEXT_MEM_BYTES" 'BEGIN { printf "%.6f", bytes / (1024.0 * 1024.0 * 1024.0) }')
ALL_THREE_MEM_GB=$(awk -v bytes="$ALL_THREE_MEM_BYTES" 'BEGIN { printf "%.6f", bytes / (1024.0 * 1024.0 * 1024.0) }')

VERDICT=$(awk \
  -v wall="$PROJECTED_WALL_H" \
  -v mem="$PROJECTED_PEAK_GB" \
  -v green="$WALL_HOURS_GREEN" \
  -v red_h="$WALL_HOURS_RED" \
  -v max_mem="$PEAK_MEM_GB" '
BEGIN {
  if (wall > red_h || mem > max_mem) {
    print "RED"
  } else if (wall > green) {
    print "YELLOW"
  } else {
    print "GREEN"
  }
}')

GREEN_LABEL=$(format_compact_number "$WALL_HOURS_GREEN")
RED_LABEL=$(format_compact_number "$WALL_HOURS_RED")
PEAK_LABEL=$(format_compact_number "$PEAK_MEM_GB")

printf 'Phase 4 budget projection from %s:\n' "$CSV_PATH"
printf '  input rows:           %s (1 seed × 1 epoch on real CWRU)\n' "$ROW_COUNT"
printf '  per-condition timing:\n'
printf '    sensors-only:       %ss, %sGB peak\n' "$(format_one_decimal "$SENSORS_WALL_S")" "$(format_one_decimal "$SENSORS_MEM_GB")"
printf '    vision+text:        %ss, %sGB peak\n' "$(format_one_decimal "$VISION_TEXT_WALL_S")" "$(format_one_decimal "$VISION_TEXT_MEM_GB")"
printf '    all-three:          %ss, %sGB peak\n' "$(format_one_decimal "$ALL_THREE_WALL_S")" "$(format_one_decimal "$ALL_THREE_MEM_GB")"
printf '  projected total:      %sh wall, %sGB peak (for %s seeds × %s epochs)\n' \
  "$(format_one_decimal "$PROJECTED_WALL_H")" \
  "$(format_one_decimal "$PROJECTED_PEAK_GB")" \
  "$SEEDS" \
  "$EPOCHS"
printf '\n'
printf 'Thresholds: GREEN ≤%sh, YELLOW %s–%sh, RED >%sh; peak ≤%sGB\n' \
  "$GREEN_LABEL" \
  "$GREEN_LABEL" \
  "$RED_LABEL" \
  "$RED_LABEL" \
  "$PEAK_LABEL"
printf 'VERDICT: %s\n' "$VERDICT"

case "$VERDICT" in
  YELLOW)
    echo "WARNING: projected wall time is in the YELLOW zone (${GREEN_LABEL}–${RED_LABEL}h). Document risk in RUNNING_NOTES.md before proceeding." >&2
    exit 0
    ;;
  RED)
    echo "Redesign options: reduce --seeds (e.g. 3), reduce --epochs (e.g. 3), simplify encoder, or drop a condition." >&2
    exit 1
    ;;
  *)
    exit 0
    ;;
esac
