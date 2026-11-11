#!/usr/bin/env bash
# Runner for the GR-1 tabletop runs (GR00T N1.6 (GR-1)): starts N_SERVERS policy servers (gr1/gr1_server.py) on one GPU,
# runs N_WORKERS closed-loop workers (gr1/gr1_closed_loop.py) per condition, one condition after another, and stops
# the servers at the end. See docs/gr1_tabletop.md.
#
# Required environment variables:
#   GR1_ENV   clone of NVIDIA/Isaac-GR00T at tag n1.6.1-release, set up with `uv sync` and
#             gr00t/eval/sim/robocasa-gr1-tabletop-tasks/setup_RoboCasaGR1TabletopTasks.sh
#   GR1_CKPT  local snapshot of nvidia/GR00T-N1.6-3B (revision d0814e7); not needed for MODE=check_scene or hold alone
#   OUT_ROOT  output directory; episodes go to $OUT_ROOT/<mechanism>/<instruction>/episodes_w*.jsonl
#   GPU       GPU index for the servers and the renderer
#   PORT0     first server port (ports PORT0 .. PORT0+N_SERVERS-1); not needed for MODE=check_scene or hold alone
# Optional: MODE (eval | calibrate | check_scene, default eval), N_SERVERS (3), N_WORKERS (12), GATE_FILE
#   (data/gr1/gr1_gate.json of this repository), STRINGS (gr1/configs/gr1_instruction_strings.json), SCENES
#   (gr1/configs/gr1_scenes.json), GR1_PY and SERVER_PY (python of the robocasa_uv and of the Isaac-GR00T environment).
#
# Examples:
#   MODE=eval bash gr1/run_gr1.sh none/original none/empty none/neutral none/nonsense none/template_only
#   MODE=eval bash gr1/run_gr1.sh none/impossible_action none/absent_object prefix/original prefix/empty prefix/nonsense
#   MODE=eval bash gr1/run_gr1.sh gate/original gate/empty gate/neutral gate/nonsense gate/template_only
#   MODE=eval bash gr1/run_gr1.sh hold
#   MODE=calibrate bash gr1/run_gr1.sh       # then: python3 gr1/gr1_gate_calib.py --in $OUT_ROOT/calibrate --out ...
#   MODE=check_scene bash gr1/run_gr1.sh
set -euo pipefail
MODE="${MODE:-eval}"
GR1_ENV="${GR1_ENV:?GR1_ENV required (Isaac-GR00T n1.6.1 clone)}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT required}"
GPU="${GPU:?GPU required (GPU index)}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
GR1_PY="${GR1_PY:-$GR1_ENV/gr00t/eval/sim/robocasa-gr1-tabletop-tasks/robocasa_uv/.venv/bin/python}"
SERVER_PY="${SERVER_PY:-$GR1_ENV/.venv/bin/python}"
N_SERVERS="${N_SERVERS:-3}"
N_WORKERS="${N_WORKERS:-12}"
GATE_FILE="${GATE_FILE:-$REPO/data/gr1/gr1_gate.json}"
STRINGS="${STRINGS:-$REPO/gr1/configs/gr1_instruction_strings.json}"
SCENES="${SCENES:-$REPO/gr1/configs/gr1_scenes.json}"

case "$MODE" in
  eval) CONDS="$*"; [ -n "$CONDS" ] || { echo "MODE=eval needs conditions as arguments" >&2; exit 1; } ;;
  calibrate) CONDS="calibrate" ;;
  check_scene) CONDS="check_scene" ;;
  *) echo "unknown MODE=$MODE (eval | calibrate | check_scene)" >&2; exit 1 ;;
esac
NEED_SERVER=1
case " $CONDS " in " hold "|" check_scene ") NEED_SERVER=0 ;; esac
if [ "$NEED_SERVER" = 1 ]; then
  GR1_CKPT="${GR1_CKPT:?GR1_CKPT required (local snapshot of nvidia/GR00T-N1.6-3B)}"
  PORT0="${PORT0:?PORT0 required}"
fi

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$GPU" MUJOCO_EGL_DEVICE_ID="$GPU"
export PYTHONHASHSEED=0   # the scene generation depends on Python's hash order; fixed so that a seed gives one scene
mkdir -p "$OUT_ROOT"

eplist() {
  python3 - "$MODE" "$SCENES" <<'EOF'
import sys
sys.path.insert(0, "gr1")
import gr1_common as G
mode, scenes = sys.argv[1], sys.argv[2]
if mode == "calibrate":
    eps = G.calibration_list()
elif mode == "check_scene":
    eps = [(t["k"], s) for t in G.TASKS for s in G.CHECK_SEEDS]
else:
    eps = G.scene_list(scenes)
print(" ".join(f"{k}:{s}" for k, s in eps))
EOF
}
EPS=($(cd "$REPO" && eplist))

if [ "$MODE" = check_scene ]; then
  ( cd "$GR1_ENV" && "$GR1_PY" "$REPO/gr1/gr1_closed_loop.py" --check-scene --eps "$(echo "${EPS[@]}" | tr ' ' ',')" \
      --out "$OUT_ROOT/check_scene" )
  exit 0
fi

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT
if [ "$NEED_SERVER" = 1 ]; then
  for s in $(seq 0 $((N_SERVERS - 1))); do
    PORT=$((PORT0 + s))
    ( cd "$GR1_ENV" && exec "$SERVER_PY" "$REPO/gr1/gr1_server.py" --port "$PORT" --model-path "$GR1_CKPT" ) \
      > "$OUT_ROOT/server_$PORT.log" 2>&1 &
    PIDS+=($!)
  done
  for s in $(seq 0 $((N_SERVERS - 1))); do
    PORT=$((PORT0 + s))
    for i in $(seq 1 120); do
      grep -q "server ready port=$PORT" "$OUT_ROOT/server_$PORT.log" 2>/dev/null && break
      kill -0 "${PIDS[$s]}" 2>/dev/null || { echo "server $PORT exited" >&2; tail -20 "$OUT_ROOT/server_$PORT.log" >&2; exit 1; }
      sleep 5
    done
    grep -q "server ready port=$PORT" "$OUT_ROOT/server_$PORT.log" || { echo "server $PORT not ready" >&2; exit 1; }
  done
fi

for COND in $CONDS; do
  OUT="$OUT_ROOT/$COND"
  mkdir -p "$OUT"
  if [ "$COND" = calibrate ]; then ARGS=(--calibrate); else ARGS=(--cond "$COND"); fi
  case "$COND" in gate/*) ARGS+=(--gate-file "$GATE_FILE") ;; esac
  case "$COND" in none/impossible_action|none/absent_object|prefix/original) ARGS+=(--strings "$STRINGS") ;; esac
  WP=()
  for w in $(seq 0 $((N_WORKERS - 1))); do
    L=""
    for i in "${!EPS[@]}"; do [ $((i % N_WORKERS)) = "$w" ] && L="$L${EPS[$i]},"; done
    [ -z "$L" ] && continue
    PORTARG=()
    [ "$NEED_SERVER" = 1 ] && PORTARG=(--port $((PORT0 + w % N_SERVERS)))
    ( cd "$GR1_ENV" && exec "$GR1_PY" "$REPO/gr1/gr1_closed_loop.py" "${ARGS[@]}" --eps "$L" ${PORTARG[@]+"${PORTARG[@]}"} \
        --out "$OUT" --worker "w$w" ) > "$OUT/worker_w$w.log" 2>&1 &
    WP+=($!)
  done
  RC=0
  for p in "${WP[@]}"; do wait "$p" || RC=$((RC + 1)); done
  N=$(cat "$OUT"/episodes_*.jsonl 2>/dev/null | wc -l | tr -d ' ')
  echo "$COND: $N of ${#EPS[@]} episodes, failed workers $RC"
  [ "$RC" = 0 ] && [ "$N" = "${#EPS[@]}" ] || exit 1
done
