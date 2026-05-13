#!/usr/bin/env bash
# Convenience wrapper for common training scenarios.
# Run from the repo root: ./docker/run_train.sh [skrl|rsl_rl|sb3] [TASK] [NUM_ENVS]

set -euo pipefail

FRAMEWORK=${1:-skrl}
TASK=${2:-Vel-Hovering}
NUM_ENVS=${3:-4096}

case "$FRAMEWORK" in
  skrl)    SERVICE=train-skrl ;;
  rsl_rl)  SERVICE=train-rsl-rl ;;
  sb3)     SERVICE=train-sb3; NUM_ENVS=${3:-512} ;;
  *)
    echo "Usage: $0 [skrl|rsl_rl|sb3] [TASK] [NUM_ENVS]"
    exit 1
    ;;
esac

echo "==> Building image (if needed)..."
docker compose -f "$(dirname "$0")/docker-compose.yml" build crazyplayground

echo "==> Starting training: framework=$FRAMEWORK  task=$TASK  num_envs=$NUM_ENVS"
TASK="$TASK" NUM_ENVS="$NUM_ENVS" \
  docker compose -f "$(dirname "$0")/docker-compose.yml" run --rm "$SERVICE"
