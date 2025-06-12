#!/bin/sh
#SBATCH -A IscrC_AdvCMT 
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="clip_msi1_training_$1"
#SBATCH --out="./sout/clip_msi1_training_$1.out"
#SBATCH --open-mode=truncate

echo "Running on nodes: $SLURM_NODELIST"
echo "Arguments passed: $@"

cd .
export WANDB_MODE=offline
module load anaconda3
module load cuda
conda init

conda activate test_env
#source activate test_env

srun python training_clip_msi1.py --max_epochs "$1"