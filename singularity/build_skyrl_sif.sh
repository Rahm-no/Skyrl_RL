#!/bin/bash
#SBATCH --job-name=build-skyrl-sif
#SBATCH --account=i20240005x
#SBATCH --partition=dev-x86
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/projects/I20240005/rnouaj/skyrl/logs/build_sif_%j.log
#SBATCH --error=/projects/I20240005/rnouaj/skyrl/logs/build_sif_%j_err.log

set -e

echo "========================================================"
echo "  Building SkyRL FSDP Singularity image"
echo "========================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Time started: $(date)"

FINAL_SIF="/projects/I20240005/rnouaj/skyrl/singularity/skyrl_fsdp.sif"
# Work entirely in /tmp (local SSD) to avoid NFS permission issues with fakeroot
BUILD_DIR="/tmp/sing_build_${SLURM_JOB_ID}"
LOCAL_DEF="$BUILD_DIR/skyrl_fsdp.def"
LOCAL_SIF="$BUILD_DIR/skyrl_fsdp.sif"

export SINGULARITY_TMPDIR="$BUILD_DIR/tmp"
export SINGULARITY_CACHEDIR="$BUILD_DIR/cache"
mkdir -p "$BUILD_DIR" "$SINGULARITY_TMPDIR" "$SINGULARITY_CACHEDIR"

# Copy definition to local disk
cp /projects/I20240005/rnouaj/skyrl/singularity/skyrl_fsdp.def "$LOCAL_DEF"

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
echo "Image: $(ls -lh $SIF_PATH)"
