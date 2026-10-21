#!/usr/bin/env bash
# Create backend virtual environments .venvs/<env> from envs/<env>.txt (uv, Python 3.11).
#
#   scripts/make_envs.sh asr score          # CUDA torch
#   CPU=1 scripts/make_envs.sh asr score    # CPU torch
#   scripts/make_envs.sh all
#
# Env vars: CPU=1 (CPU torch), SKIP_EXTRAS=1, STRICT=1 (extras failures are fatal),
# RECREATE=1, PYTHON=3.11. Directive format: see envs/README.md.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVS_DIR="$ROOT/envs"
PYTHON="${PYTHON:-3.11}"
BUILD_CONS="$ENVS_DIR/build.constraints.txt"  # build-time pins for sdist-only packages
command -v uv >/dev/null || { echo "uv not found on PATH" >&2; exit 1; }

envs=("$@")
[ ${#envs[@]} -eq 0 ] && envs=(asr)
if [ "${envs[0]}" = "all" ]; then
  envs=()
  for f in "$ENVS_DIR"/*.txt; do b="$(basename "$f" .txt)"; [[ "$b" == *[+.]* ]] || envs+=("$b"); done
fi

venv_python() { if [ -x "$1/Scripts/python.exe" ]; then echo "$1/Scripts/python.exe"; else echo "$1/bin/python"; fi; }

directive() { grep -E "^\s*#\s*@$2\s" "$1" | sed -E "s/^\s*#\s*@$2\s+//" || true; }

install_file() {  # env py file constraints
  local env="$1" py="$2" file="$3" cons="$4" src="$ROOT/.venvs/$1/src"
  while read -r name url ref rec; do
    [ -z "${name:-}" ] && continue
    local dst="$src/$name"
    if [ ! -d "$dst" ]; then
      mkdir -p "$src"
      if [[ "$ref" =~ ^[0-9a-f]{40}$ ]]; then
        git clone "$url" "$dst" && git -C "$dst" checkout "$ref"
        [ "${rec:-}" = "--recursive" ] && git -C "$dst" submodule update --init --recursive
      else
        if [ "${rec:-}" = "--recursive" ]; then git clone --depth 1 --branch "$ref" --recurse-submodules "$url" "$dst"
        else git clone --depth 1 --branch "$ref" "$url" "$dst"; fi
      fi
    fi
  done < <(directive "$file" git)
  local cargs=(); [ -n "$cons" ] && cargs=(-c "$cons")
  uv pip install --python "$py" -r "$file" --build-constraint "$BUILD_CONS" "${cargs[@]}"
  while read -r name rfile; do
    [ -z "${name:-}" ] && continue
    uv pip install --python "$py" -r "$src/$name/$rfile" --build-constraint "$BUILD_CONS" "${cargs[@]}"
  done < <(directive "$file" req-from)
  while read -r req; do
    [ -z "${req:-}" ] && continue
    uv pip install --python "$py" --no-deps --build-constraint "$BUILD_CONS" $req
  done < <(directive "$file" nodeps)
}

for env in "${envs[@]}"; do
  req="$ENVS_DIR/$env.txt"
  [ -f "$req" ] || { echo "no requirements file $req" >&2; exit 1; }
  if ! grep -qE '^\s*[^#[:space:]]|^\s*#\s*@' "$req"; then echo "$env: nothing to install"; continue; fi
  venv="$ROOT/.venvs/$env"
  [ "${RECREATE:-0}" = "1" ] && rm -rf "$venv"
  echo "== env $env ($([ "${CPU:-0}" = "1" ] && echo CPU || echo CUDA))"
  if [ ! -f "$venv/pyvenv.cfg" ]; then rm -rf "$venv"; uv venv "$venv" --python "$PYTHON"; fi
  py="$(venv_python "$venv")"
  torch="$(directive "$req" torch | head -n1)"
  cuda="$(directive "$req" cuda | head -n1)"; cuda="${cuda:-cu128}"
  cons=""
  if [ -n "$torch" ]; then
    index="https://download.pytorch.org/whl/$([ "${CPU:-0}" = "1" ] && echo cpu || echo "$cuda")"
    # shellcheck disable=SC2086
    uv pip install --python "$py" --index-url "$index" $torch
    cons="$venv/torch-constraints.txt"; printf '%s\n' $torch > "$cons"
  fi
  install_file "$env" "$py" "$req" "$cons"
  failed=()
  if [ "${SKIP_EXTRAS:-0}" != "1" ]; then
    for x in "$ENVS_DIR/$env+"*.txt; do
      [ -f "$x" ] || continue
      echo "   extra $(basename "$x")"
      if ! install_file "$env" "$py" "$x" "$cons"; then
        [ "${STRICT:-0}" = "1" ] && exit 1
        echo "WARNING: extra $(basename "$x") failed" >&2; failed+=("$(basename "$x")")
      fi
    done
  fi
  printf '{"env": "%s", "cpu": %s, "torch": "%s", "extras_failed": "%s"}\n' "$env" "$([ "${CPU:-0}" = "1" ] && echo true || echo false)" "$torch" "${failed[*]:-}" > "$venv/vndub_env.json"
  echo "$env: ok"
done
