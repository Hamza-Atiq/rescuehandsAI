#!/usr/bin/env bash
# Make MuJoCo draw camera images on the Kaggle T4 instead of the CPU.
#
# Kaggle mounts NVIDIA's compute libraries (CUDA) but not its EGL graphics
# libraries, so EGL falls back to a software renderer: ~150 ms per camera,
# 89% of data-generation time. This extracts the EGL libraries that exactly
# match the running driver into .cache/nvegl; kaggle_pipeline.py uses them
# automatically when that folder exists.
#
#   bash training/kaggle_gpu_render.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/.cache/nvegl"
VER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | tr -d ' ')"
echo "driver $VER"
mkdir -p "$OUT" && cd "$OUT"

RUN="NVIDIA-Linux-x86_64-$VER.run"
if [ ! -d "NVIDIA-Linux-x86_64-$VER" ]; then
  wget -q "https://us.download.nvidia.com/tesla/$VER/$RUN" \
    || wget -q "https://download.nvidia.com/XFree86/Linux-x86_64/$VER/$RUN"
  sh "$RUN" --extract-only > /dev/null
  rm -f "$RUN"
fi
SRC="NVIDIA-Linux-x86_64-$VER"
for lib in libEGL_nvidia libnvidia-eglcore libnvidia-glsi libnvidia-gpucomp libnvidia-glcore libnvidia-tls; do
  [ -f "$SRC/$lib.so.$VER" ] && cp -f "$SRC/$lib.so.$VER" .
done
ln -sf "libEGL_nvidia.so.$VER" libEGL_nvidia.so.0
cat > 10_nvidia.json <<EOF
{"file_format_version": "1.0.0", "ICD": {"library_path": "$OUT/libEGL_nvidia.so.0"}}
EOF

echo "missing dependencies (should be empty):"
LD_LIBRARY_PATH="$OUT" ldd libEGL_nvidia.so.0 | grep "not found" || true

cd "$ROOT"
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=src \
  __EGL_VENDOR_LIBRARY_FILENAMES="$OUT/10_nvidia.json" LD_LIBRARY_PATH="$OUT:${LD_LIBRARY_PATH:-}" \
  .venv-train/bin/python scripts/profile_generation.py --steps 60 2>&1 | grep -v -E "wrapt|sitecustomize"
