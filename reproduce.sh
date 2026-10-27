#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image_name="${REPRO_IMAGE:-climateccai-locked-v4:latest}"
artifact_dir="${REPRO_OUTPUT_DIR:-${repo_root}/reproduction-output}"

for input in readymix.jsonl readymix.state.json; do
  if [[ ! -f "${repo_root}/data/raw/ec3_materials/${input}" ]]; then
    echo "error: missing licensed input: data/raw/ec3_materials/${input}" >&2
    echo "See docs/DATA_ACCESS.md for the frozen snapshot and hash requirements." >&2
    exit 2
  fi
done
if [[ ! -f "${repo_root}/data/processed/clf_concrete_a1a3.csv" ]]; then
  echo "error: missing portfolio input: data/processed/clf_concrete_a1a3.csv" >&2
  echo "See docs/DATA_ACCESS.md for the required CLF cache." >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "error: Docker is required for the locked reproduction." >&2
  exit 2
fi

mkdir -p "${artifact_dir}"
if find "${artifact_dir}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  echo "error: artifact directory is not empty: ${artifact_dir}" >&2
  echo "set REPRO_OUTPUT_DIR to a new or empty directory" >&2
  exit 2
fi

docker build --platform linux/amd64 --tag "${image_name}" "${repo_root}"
docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --volume "${artifact_dir}:/artifacts" \
  "${image_name}"

echo "Reproduction artifacts: ${artifact_dir}"
