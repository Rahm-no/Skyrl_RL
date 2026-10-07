#!/bin/bash
# Submit from the project root: sbatch singularity/build_skyrl_sif.sh
# Override output path via env vars, e.g.:
#   SKYRL_SIF=/my/path/skyrl_fsdp.sif sbatch singularity/build_skyrl_sif.sh

#SBATCH --job-name=build-skyrl-sif
#SBATCH --account=i20240005x
#SBATCH --partition=dev-x86
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/build_sif_%j.log
#SBATCH --error=logs/build_sif_%j_err.log

set -e

# ── Configurable paths (override via environment variables) ──────────────────
PROJECT="${SKYRL_PROJECT:-/projects/I20240005/rnouaj/skyrl}"
FINAL_SIF="${SKYRL_SIF:-$PROJECT/singularity/skyrl_fsdp.sif}"
# ─────────────────────────────────────────────────────────────────────────────

echo "========================================================"
echo "  Building SkyRL FSDP Singularity image"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Time started: $(date)"

# Work entirely in /tmp (local SSD) to avoid NFS permission issues with fakeroot
BUILD_DIR="/tmp/sing_build_${SLURM_JOB_ID}"
LOCAL_DEF="$BUILD_DIR/skyrl_fsdp.def"
LOCAL_SIF="$BUILD_DIR/skyrl_fsdp.sif"

export SINGULARITY_TMPDIR="$BUILD_DIR/tmp"
export SINGULARITY_CACHEDIR="$BUILD_DIR/cache"
mkdir -p "$BUILD_DIR" "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

cp "$PROJECT/singularity/skyrl_fsdp.def" "$LOCAL_DEF"

echo "Building from: $LOCAL_DEF  (local copy)"
echo "Temp output:   $LOCAL_SIF"
echo "Final output:  $FINAL_SIF"
echo ""

singularity build --fakeroot "$LOCAL_SIF" "$LOCAL_DEF" 2>&1

echo ""
echo "Copying SIF to projects filesystem..."
cp "$LOCAL_SIF" "$FINAL_SIF"
echo "Done."

echo ""
echo "Build complete at: $(date)"
echo "Image: $(ls -lh "$FINAL_SIF")"
