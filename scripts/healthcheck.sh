#!/usr/bin/env bash
# ============================================================
# healthcheck.sh — Service Health Status Script
# Smart Factory Predictive Maintenance Platform
#
# Usage:
#   bash scripts/healthcheck.sh            # check all services
#   bash scripts/healthcheck.sh --json     # JSON output for monitoring
#
# Exit codes:
#   0 — all services healthy
#   1 — one or more services unhealthy or unreachable
# ============================================================

set -euo pipefail

# ── Configuration (override via environment) ─────────────────
BACKEND_URL="${BACKEND_URL:-http://localhost:5000}"
MLFLOW_URL="${MLFLOW_URL:-http://localhost:5001}"
FRONTEND_URL="${FRONTEND_URL:-http://localhost:80}"
TIMEOUT="${HEALTH_CHECK_TIMEOUT:-5}"
JSON_OUTPUT=false

# ── Argument parsing ─────────────────────────────────────────
for arg in "$@"; do
  case $arg in
    --json) JSON_OUTPUT=true ;;
    *) ;;
  esac
done

# ── Color codes ──────────────────────────────────────────────
if [ -t 1 ] && [ "$JSON_OUTPUT" = "false" ]; then
  GREEN='\033[0;32m'
  RED='\033[0;31m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  RESET='\033[0m'
else
  GREEN='' RED='' YELLOW='' BOLD='' RESET=''
fi

# ── Helper Functions ─────────────────────────────────────────
check_service() {
  local name="$1"
  local url="$2"
  local response http_code body

  # Use curl to get status code and body simultaneously
  response=$(curl -sf --max-time "$TIMEOUT" -w "\n__HTTP_CODE__%{http_code}" "$url" 2>/dev/null || true)
  http_code=$(echo "$response" | grep '__HTTP_CODE__' | sed 's/__HTTP_CODE__//')
  body=$(echo "$response" | grep -v '__HTTP_CODE__' || true)

  if [ -z "$http_code" ]; then
    echo "unreachable"
    return 1
  elif [ "$http_code" -ge 200 ] && [ "$http_code" -lt 300 ]; then
    echo "healthy"
    return 0
  else
    echo "unhealthy (HTTP $http_code)"
    return 1
  fi
}

# ── Main Health Check ────────────────────────────────────────
OVERALL_STATUS=0
declare -A SERVICE_RESULTS

SERVICES=(
  "Backend API:${BACKEND_URL}/health"
  "MLflow Server:${MLFLOW_URL}/health"
  "Frontend (Nginx):${FRONTEND_URL}/nginx-health"
)

for service_entry in "${SERVICES[@]}"; do
  name="${service_entry%%:*}"
  url="${service_entry#*:}"
  status=$(check_service "$name" "$url") || OVERALL_STATUS=1
  SERVICE_RESULTS["$name"]="$status"
done

# ── Output ───────────────────────────────────────────────────
if [ "$JSON_OUTPUT" = "true" ]; then
  # Machine-readable JSON output
  printf '{\n'
  printf '  "timestamp": "%s",\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '  "overall": "%s",\n' "$([ $OVERALL_STATUS -eq 0 ] && echo healthy || echo degraded)"
  printf '  "services": {\n'
  first=true
  for name in "${!SERVICE_RESULTS[@]}"; do
    [ "$first" = "true" ] || printf ',\n'
    printf '    "%s": "%s"' "$name" "${SERVICE_RESULTS[$name]}"
    first=false
  done
  printf '\n  }\n'
  printf '}\n'
else
  # Human-readable terminal output
  echo ""
  echo "${BOLD}╔══════════════════════════════════════════════════════╗${RESET}"
  echo "${BOLD}║   Smart Factory PDM — Service Health Check           ║${RESET}"
  echo "${BOLD}╚══════════════════════════════════════════════════════╝${RESET}"
  echo ""
  echo "  Timestamp : $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo ""

  for name in "${!SERVICE_RESULTS[@]}"; do
    status="${SERVICE_RESULTS[$name]}"
    if [ "$status" = "healthy" ]; then
      icon="${GREEN}✔${RESET}"
      color="${GREEN}"
    else
      icon="${RED}✘${RESET}"
      color="${RED}"
    fi
    printf "  %b  %-22s %b%s%b\n" "$icon" "$name" "$color" "$status" "$RESET"
  done

  echo ""
  if [ $OVERALL_STATUS -eq 0 ]; then
    echo "  ${GREEN}${BOLD}Overall: ALL SERVICES HEALTHY${RESET}"
  else
    echo "  ${RED}${BOLD}Overall: ONE OR MORE SERVICES DEGRADED${RESET}"
    echo ""
    echo "  Troubleshooting:"
    echo "    docker-compose ps         # Check container status"
    echo "    docker-compose logs -f    # Follow all service logs"
  fi
  echo ""
fi

exit $OVERALL_STATUS
