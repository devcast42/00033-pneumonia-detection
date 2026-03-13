#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
API_HOST="127.0.0.1"
API_PORT="8002"
APP_HOST="127.0.0.1"
APP_PORT="3000"

kill_port() {
  local port="$1"
  local pids
  pids="$(lsof -ti tcp:"$port" || true)"
  if [[ -n "${pids}" ]]; then
    echo "${pids}" | xargs kill -9 || true
  fi
}

cleanup() {
  if [[ -n "${API_PID:-}" ]]; then
    kill "${API_PID}" 2>/dev/null || true
  fi
  if [[ -n "${APP_PID:-}" ]]; then
    kill "${APP_PID}" 2>/dev/null || true
  fi
  kill_port "${API_PORT}"
  kill_port "${APP_PORT}"
  wait 2>/dev/null || true
  echo "Puertos liberados: ${APP_PORT} y ${API_PORT}"
}

if [[ "${1:-}" == "--stop" ]]; then
  kill_port "${API_PORT}"
  kill_port "${APP_PORT}"
  echo "Puertos liberados: ${APP_PORT} y ${API_PORT}"
  exit 0
fi

trap cleanup EXIT INT TERM

kill_port "${API_PORT}"
kill_port "${APP_PORT}"

cd "${ROOT_DIR}"
python3 -m uvicorn project.api.main:app --host "${API_HOST}" --port "${API_PORT}" &
API_PID=$!

cd "${ROOT_DIR}/app"
npm run dev -- --hostname "${APP_HOST}" --port "${APP_PORT}" &
APP_PID=$!

echo "Frontend: http://${APP_HOST}:${APP_PORT}"
echo "API: http://${API_HOST}:${API_PORT}"
echo "Detener todo: Ctrl+C"

wait
