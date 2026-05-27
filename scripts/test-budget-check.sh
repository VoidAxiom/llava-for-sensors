#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUDGET_CHECK="$SCRIPT_DIR/budget-check.sh"

if [ ! -f "$BUDGET_CHECK" ] || [ ! -x "$BUDGET_CHECK" ]; then
  echo "ERROR: budget-check.sh is not found or executable: $BUDGET_CHECK" >&2
  exit 2
fi

TEST_TMP="$(mktemp -d)"
trap 'rm -rf "$TEST_TMP"' EXIT

PASS=0
FAIL=0

run_test() {
  local name="$1"
  local result="$2"
  local expected="$3"

  if [[ "$result" == "$expected" ]]; then
    echo "PASS: $name"
    PASS=$((PASS + 1))
  else
    echo "FAIL: $name (got '$result', expected '$expected')"
    FAIL=$((FAIL + 1))
  fi
}

pass_test() {
  local name="$1"
  run_test "$name" "PASS" "PASS"
}

fail_test() {
  local name="$1"
  local detail="$2"
  echo "FAIL: $name ($detail)"
  FAIL=$((FAIL + 1))
}

contains_text() {
  local haystack="$1"
  local needle="$2"

  case "$haystack" in
    *"$needle"*) return 0 ;;
    *) return 1 ;;
  esac
}

CSV_GREEN="$TEST_TMP/green.csv"
cat > "$CSV_GREEN" <<'CSV'
condition,seed,final_val_f1,wall_time_s,peak_memory_bytes
sensors-only,0,0.85,600,8589934592
vision+text,0,0.87,700,8589934592
all-three,0,0.90,800,8589934592
CSV

set +e
OUTPUT=$(bash "$BUDGET_CHECK" "$CSV_GREEN" 2>"$TEST_TMP/green.err")
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" -eq 0 ] && contains_text "$OUTPUT" "VERDICT: GREEN"; then
  pass_test "GREEN verdict (wall=14.6h)"
else
  fail_test "GREEN verdict (wall=14.6h)" "exit=$EXIT_CODE stdout='$OUTPUT'"
fi

CSV_YELLOW="$TEST_TMP/yellow.csv"
cat > "$CSV_YELLOW" <<'CSV'
condition,seed,final_val_f1,wall_time_s,peak_memory_bytes
sensors-only,0,0.85,900,8589934592
vision+text,0,0.87,1000,8589934592
all-three,0,0.90,1100,8589934592
CSV

set +e
OUTPUT=$(bash "$BUDGET_CHECK" "$CSV_YELLOW" 2>"$TEST_TMP/yellow.err")
EXIT_CODE=$?
set -e
STDERR_OUT=$(sed -n '1,20p' "$TEST_TMP/yellow.err")
if [ "$EXIT_CODE" -eq 0 ] && contains_text "$OUTPUT" "VERDICT: YELLOW" && contains_text "$STDERR_OUT" "WARNING"; then
  pass_test "YELLOW verdict (wall=20.8h)"
else
  fail_test "YELLOW verdict (wall=20.8h)" "exit=$EXIT_CODE stdout='$OUTPUT' stderr='$STDERR_OUT'"
fi

CSV_RED_WALL="$TEST_TMP/red-wall.csv"
cat > "$CSV_RED_WALL" <<'CSV'
condition,seed,final_val_f1,wall_time_s,peak_memory_bytes
sensors-only,0,0.85,1500,8589934592
vision+text,0,0.87,1500,8589934592
all-three,0,0.90,1500,8589934592
CSV

set +e
OUTPUT=$(bash "$BUDGET_CHECK" "$CSV_RED_WALL" 2>"$TEST_TMP/red-wall.err")
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" -eq 1 ] && contains_text "$OUTPUT" "VERDICT: RED"; then
  pass_test "RED verdict by wall time (wall=31.2h)"
else
  fail_test "RED verdict by wall time (wall=31.2h)" "exit=$EXIT_CODE stdout='$OUTPUT'"
fi

CSV_RED_MEM="$TEST_TMP/red-memory.csv"
cat > "$CSV_RED_MEM" <<'CSV'
condition,seed,final_val_f1,wall_time_s,peak_memory_bytes
sensors-only,0,0.85,600,53687091200
vision+text,0,0.87,700,53687091200
all-three,0,0.90,800,53687091200
CSV

set +e
OUTPUT=$(bash "$BUDGET_CHECK" "$CSV_RED_MEM" 2>"$TEST_TMP/red-memory.err")
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" -eq 1 ] && contains_text "$OUTPUT" "VERDICT: RED"; then
  pass_test "RED verdict by memory (peak=50.0GB)"
else
  fail_test "RED verdict by memory (peak=50.0GB)" "exit=$EXIT_CODE stdout='$OUTPUT'"
fi

BAD_CSV="$TEST_TMP/bad-header.csv"
cat > "$BAD_CSV" <<'CSV'
col1,col2,col3
CSV

set +e
STDERR_OUT=$(bash "$BUDGET_CHECK" "$BAD_CSV" 2>&1 1>/dev/null)
EXIT_CODE=$?
set -e
if [ "$EXIT_CODE" -ne 0 ] && printf '%s' "$STDERR_OUT" | grep -Eiq 'ERROR|header'; then
  pass_test "CSV validation failure (wrong header)"
else
  fail_test "CSV validation failure (wrong header)" "exit=$EXIT_CODE stderr='$STDERR_OUT'"
fi

CSV_INCOMPLETE="$TEST_TMP/incomplete.csv"
cat > "$CSV_INCOMPLETE" <<'CSV'
condition,seed,final_val_f1,wall_time_s,peak_memory_bytes
sensors-only,0,0.85,600,8589934592
CSV

set +e
OUTPUT=$(bash "$BUDGET_CHECK" "$CSV_INCOMPLETE" 2>"$TEST_TMP/incomplete.err")
EXIT_CODE=$?
set -e
STDERR_OUT=$(sed -n '1,20p' "$TEST_TMP/incomplete.err")
if [ "$EXIT_CODE" -eq 1 ] && contains_text "$STDERR_OUT" "missing required condition row(s)" && contains_text "$STDERR_OUT" "vision+text" && contains_text "$STDERR_OUT" "all-three"; then
  pass_test "CSV validation failure (missing condition rows)"
else
  fail_test "CSV validation failure (missing condition rows)" "exit=$EXIT_CODE stdout='$OUTPUT' stderr='$STDERR_OUT'"
fi

echo "Summary: $PASS passed, $FAIL failed."

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi

exit 0
