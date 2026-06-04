#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_ROOT="${TRIPOSPLAT_SOURCE_ROOT:-${ROOT_DIR}/tmp/upstream/TripoSplat/VAST-AI-TripoSplat}"
OUTPUT_ROOT="${TRIPOSPLAT_OUTPUT_ROOT:-${ROOT_DIR}/crates/burn_triposplat/assets/models/TripoSplat}"
PRECISION="${TRIPOSPLAT_PRECISION:-f16}"
PART_SIZE_MIB="${TRIPOSPLAT_PART_SIZE_MIB:-64}"
DOWNLOAD=1
WRITE_PARTS=1
OVERWRITE=0
OVERWRITE_PARTS=0
VALIDATE_ONLY=0

usage() {
  cat <<'EOF'
Usage: scripts/triposplat_bootstrap.sh [options]

Download or reuse official TripoSplat upstream safetensors, then import them
into repo-canonical BurnPack artifacts and optional wasm parts.

Options:
  --source-root PATH      Upstream safetensors root. Default:
                          tmp/upstream/TripoSplat/VAST-AI-TripoSplat
  --output-root PATH      BurnPack output root. Default:
                          crates/burn_triposplat/assets/models/TripoSplat
  --precision f16|f32|both
                          BurnPack precision to create. Default: f16.
                          Import uses the WGPU backend because official
                          TripoSplat flow/decoder sources contain F16 tensors.
  --part-size-mib N       Wasm part size in MiB. Default: 64
  --skip-download         Do not download from Hugging Face; validate source root only.
  --no-parts              Do not generate .bpk.parts.json manifests.
  --overwrite             Replace existing .bpk files.
  --overwrite-parts       Replace existing .bpk.parts.json and part files.
  --validate-only         Validate source/output paths; do not import.
  -h, --help              Show this help.

Environment overrides:
  TRIPOSPLAT_SOURCE_ROOT, TRIPOSPLAT_OUTPUT_ROOT, TRIPOSPLAT_PRECISION,
  TRIPOSPLAT_PART_SIZE_MIB
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --precision)
      PRECISION="$2"
      shift 2
      ;;
    --part-size-mib)
      PART_SIZE_MIB="$2"
      shift 2
      ;;
    --skip-download)
      DOWNLOAD=0
      shift
      ;;
    --no-parts)
      WRITE_PARTS=0
      shift
      ;;
    --overwrite)
      OVERWRITE=1
      shift
      ;;
    --overwrite-parts)
      OVERWRITE_PARTS=1
      shift
      ;;
    --validate-only)
      VALIDATE_ONLY=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "[triposplat-bootstrap] unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "${PRECISION}" in
  f16 | f32 | both) ;;
  *)
    echo "[triposplat-bootstrap] --precision must be f16, f32, or both (got '${PRECISION}')" >&2
    exit 2
    ;;
esac

required_sources=(
  "background_removal/birefnet.safetensors"
  "diffusion_models/triposplat_fp16.safetensors"
  "clip_vision/dino_v3_vit_h.safetensors"
  "vae/flux2-vae.safetensors"
  "vae/triposplat_vae_decoder_fp16.safetensors"
)

missing_sources() {
  local missing=0
  local rel
  for rel in "${required_sources[@]}"; do
    if [[ ! -f "${SOURCE_ROOT}/${rel}" ]]; then
      echo "${rel}"
      missing=1
    fi
  done
  return "${missing}"
}

download_hf_snapshot() {
  mkdir -p "${SOURCE_ROOT}"
  if command -v hf >/dev/null 2>&1; then
    echo "[triposplat-bootstrap] downloading VAST-AI/TripoSplat with Hugging Face CLI"
    hf download VAST-AI/TripoSplat --local-dir "${SOURCE_ROOT}"
    return
  fi
  if python3 -c "import huggingface_hub" >/dev/null 2>&1; then
    echo "[triposplat-bootstrap] downloading VAST-AI/TripoSplat with huggingface_hub"
    python3 - "${SOURCE_ROOT}" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(repo_id="VAST-AI/TripoSplat", local_dir=sys.argv[1])
PY
    return
  fi
  cat >&2 <<'EOF'
[triposplat-bootstrap] cannot download VAST-AI/TripoSplat automatically.
Install one of:
  pip install -U "huggingface_hub[cli]"
  pip install -U huggingface_hub
or rerun with --skip-download after placing files under --source-root.
EOF
  exit 1
}

if ! missing="$(missing_sources)"; then
  if [[ "${DOWNLOAD}" == "1" ]]; then
    echo "[triposplat-bootstrap] missing upstream files under ${SOURCE_ROOT}:"
    echo "${missing}" | sed 's/^/[triposplat-bootstrap]   /'
    download_hf_snapshot
  fi
fi

if ! missing="$(missing_sources)"; then
  echo "[triposplat-bootstrap] source root is still incomplete: ${SOURCE_ROOT}" >&2
  echo "${missing}" | sed 's/^/[triposplat-bootstrap]   missing: /' >&2
  exit 1
fi

run_import() {
  local precision="$1"
  local import_features="import,backend_wgpu"
  local args=(
    run
    -p burn_triposplat
    --bin triposplat_import
    --features "${import_features}"
    --
    --source-root "${SOURCE_ROOT}"
    --output-root "${OUTPUT_ROOT}"
    --precision "${precision}"
    --part-size-mib "${PART_SIZE_MIB}"
  )
  if [[ "${VALIDATE_ONLY}" == "1" ]]; then
    args+=(--validate-only)
  fi
  if [[ "${OVERWRITE}" == "1" ]]; then
    args+=(--overwrite)
  fi
  if [[ "${WRITE_PARTS}" == "1" ]]; then
    args+=(--parts)
  fi
  if [[ "${OVERWRITE_PARTS}" == "1" ]]; then
    args+=(--overwrite-parts)
  fi

  echo "[triposplat-bootstrap] import precision=${precision}"
  cargo "${args[@]}"
}

case "${PRECISION}" in
  both)
    run_import f16
    run_import f32
    ;;
  *)
    run_import "${PRECISION}"
    ;;
esac

echo "[triposplat-bootstrap] done"
