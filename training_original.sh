#!/bin/bash
#SBATCH -A IscrC_AdvCMT 
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="clip_msi2"
#SBATCH --out="./sout/tmp.out"
#SBATCH --open-mode=truncate


timestamp=$(date +%Y%m%d_%H%M%S)
outfile="./sout/clip_msi2_original_training_${SLURM_JOB_ID}_${timestamp}.out"
exec > >(tee -a "$outfile") 2>&1

echo "Running on nodes: $SLURM_NODELIST"

pwd
mkdir -p sout
ls -l
export WANDB_MODE=offline
module load anaconda3
module load cuda

source activate test_env

srun python training_clip2_original.py