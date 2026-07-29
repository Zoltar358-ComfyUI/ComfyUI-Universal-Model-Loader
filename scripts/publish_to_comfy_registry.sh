#!/usr/bin/env bash
set -Eeuo pipefail

# Manual Comfy Registry publisher for ComfyUI-Universal-Model-Loader.
# Based on https://docs.comfy.org/registry/publishing
#
# Recommended use:
#   ./scripts/publish_to_comfy_registry.sh
#
# Check-only mode, no publish:
#   ./scripts/publish_to_comfy_registry.sh --check-only
#
# Optional changelog:
#   ./scripts/publish_to_comfy_registry.sh --changelog "Initial 1.0.0 release"
#
# Optional token environment mode:
#   REGISTRY_ACCESS_TOKEN='pat-...' ./scripts/publish_to_comfy_registry.sh --token-env
#
# Notes:
# - The safest/default mode lets `comfy node publish` prompt for the API key.
# - Create the API key from https://registry.comfy.org/nodes for publisher @Zoltar358.
# - Paste the key with right-click paste if possible; the Comfy docs warn Ctrl+V can add a hidden \x16 character on some systems.
# - This script never writes the Registry token to disk.

usage() {
  cat <<'EOF'
Usage: scripts/publish_to_comfy_registry.sh [OPTIONS]

Options:
  --check-only             Run metadata/git/comfy validation only; do not publish.
  --token-env              Use REGISTRY_ACCESS_TOKEN from the environment instead of the hidden comfy prompt.
  --changelog TEXT         Pass changelog text to the Registry publish command.
  --changelog-file PATH    Pass changelog file to the Registry publish command.
  -h, --help               Show this help.

Default publish mode:
  Runs `comfy node publish` and lets comfy-cli prompt for the Registry API key.

Examples:
  scripts/publish_to_comfy_registry.sh --check-only
  scripts/publish_to_comfy_registry.sh --changelog "Initial 1.0.0 release"
  REGISTRY_ACCESS_TOKEN='pat-...' scripts/publish_to_comfy_registry.sh --token-env
EOF
}

CHECK_ONLY=0
USE_TOKEN_ENV=0
CHANGELOG=""
CHANGELOG_FILE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --token-env)
      USE_TOKEN_ENV=1
      shift
      ;;
    --changelog)
      [[ $# -ge 2 ]] || { echo "error: --changelog needs text" >&2; exit 2; }
      CHANGELOG="$2"
      shift 2
      ;;
    --changelog-file)
      [[ $# -ge 2 ]] || { echo "error: --changelog-file needs a path" >&2; exit 2; }
      CHANGELOG_FILE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$CHANGELOG" && -n "$CHANGELOG_FILE" ]]; then
  echo "error: --changelog and --changelog-file are mutually exclusive" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

COMFY_ENV=(env -u PYTHONPATH -u VIRTUAL_ENV -u PYTHONHOME)

if ! command -v comfy >/dev/null 2>&1; then
  cat >&2 <<'EOF'
error: comfy-cli was not found on PATH.
Install it first, for example:
  uv tool install comfy-cli
EOF
  exit 1
fi

if [[ ! -f pyproject.toml ]]; then
  echo "error: pyproject.toml not found in $REPO_ROOT" >&2
  exit 1
fi

if [[ ! -d .git ]]; then
  echo "error: $REPO_ROOT is not a git repository" >&2
  exit 1
fi

echo "==> Repository"
printf '    %s\n' "$REPO_ROOT"

echo "==> Registry metadata"
python3 - <<'PY'
import pathlib, re, sys, tomllib
path = pathlib.Path('pyproject.toml')
data = tomllib.loads(path.read_text())
project = data.get('project', {})
comfy = data.get('tool', {}).get('comfy', {})
name = project.get('name', '')
version = project.get('version', '')
publisher = comfy.get('PublisherId', '')
display = comfy.get('DisplayName', '')
repo = project.get('urls', {}).get('Repository', '')
errors = []
if not name:
    errors.append('project.name is required')
if not re.fullmatch(r'\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?', version):
    errors.append(f'project.version must be semantic versioning, got {version!r}')
if not publisher:
    errors.append('tool.comfy.PublisherId is required')
if publisher.startswith('@'):
    errors.append('tool.comfy.PublisherId must not include @')
if not repo.startswith('https://github.com/'):
    errors.append('project.urls.Repository should be a public GitHub URL')
print(f'    project.name       = {name}')
print(f'    project.version    = {version}')
print(f'    PublisherId        = {publisher}')
print(f'    DisplayName        = {display}')
print(f'    Repository         = {repo}')
if errors:
    print('\nmetadata errors:', file=sys.stderr)
    for error in errors:
        print(f'  - {error}', file=sys.stderr)
    sys.exit(1)
PY

echo "==> Git state"
git status --short --branch
if [[ -n "$(git status --porcelain)" ]]; then
  cat >&2 <<'EOF'
error: git working tree is not clean.
Commit or stash changes before publishing so the Registry package matches GitHub.
EOF
  exit 1
fi

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "main" ]]; then
  echo "warning: current branch is '$current_branch', not 'main'" >&2
fi

echo "==> Comfy CLI validation"
"${COMFY_ENV[@]}" comfy node validate

if [[ "$CHECK_ONLY" -eq 1 ]]; then
  echo "==> Check-only mode complete; publish was not attempted."
  exit 0
fi

publish_args=(node publish)
if [[ -n "$CHANGELOG" ]]; then
  publish_args+=(--changelog "$CHANGELOG")
elif [[ -n "$CHANGELOG_FILE" ]]; then
  publish_args+=(--changelog-file "$CHANGELOG_FILE")
fi

if [[ "$USE_TOKEN_ENV" -eq 1 ]]; then
  if [[ -z "${REGISTRY_ACCESS_TOKEN:-}" ]]; then
    echo "error: --token-env requires REGISTRY_ACCESS_TOKEN to be set" >&2
    exit 2
  fi
  # Strip whitespace, CR/LF, and the Ctrl+V \x16 character mentioned in the Comfy docs.
  token="$(printf '%s' "$REGISTRY_ACCESS_TOKEN" | tr -d '\r\n[:space:]\026')"
  if [[ -z "$token" ]]; then
    echo "error: REGISTRY_ACCESS_TOKEN became empty after sanitizing" >&2
    exit 2
  fi
  echo "==> Publishing with token from REGISTRY_ACCESS_TOKEN"
  "${COMFY_ENV[@]}" comfy "${publish_args[@]}" --token "$token"
else
  cat <<'EOF'
==> Publishing with comfy-cli hidden prompt
When prompted, paste the Registry Publishing API key for publisher @Zoltar358.
Recommended by Comfy docs: right-click paste to avoid adding a hidden Ctrl+V \x16 character.
EOF
  "${COMFY_ENV[@]}" comfy "${publish_args[@]}"
fi

echo "==> Publish command completed."
