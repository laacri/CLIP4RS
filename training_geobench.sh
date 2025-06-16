#!/bin/bash
#SBATCH -A IscrC_AdvCMT 
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name="clip_msi"
#SBATCH --out="./sout/tmp.out"
#SBATCH --open-mode=truncate

timestamp=$(date +%Y%m%d_%H%M%S)
#outfile="./sout/clip_msi${2}_geobench_brick_${SLURM_JOB_ID}_${timestamp}.out"
#outfile="./sout/clip_msi${2}_geobench_pv4ger_${SLURM_JOB_ID}_${timestamp}.out"
#outfile="./sout/clip_msi${2}_geobench_forestnet_${SLURM_JOB_ID}_${timestamp}.out"
#outfile="./sout/clip_msi${2}_geobench_eurosat_${SLURM_JOB_ID}_${timestamp}.out"
outfile="./sout/clip_msi${2}_geobench_so2sat_${SLURM_JOB_ID}_${timestamp}.out"
exec > >(tee -a "$outfile") 2>&1

echo "Running on nodes: $SLURM_NODELIST"
echo "Arguments passed: $1 $2"

pwd
mkdir -p sout
ls -l
export WANDB_MODE=offline
module load anaconda3
module load cuda

source activate test_env

#srun python training_geobench_brick.py --max_epochs "$1" --model "$2"
#srun python training_geobench_pv4ger.py --max_epochs "$1" --model "$2"
#srun python training_geobench_forestnet.py --max_epochs "$1" --model "$2"
#srun python training_geobench_eurosat.py --max_epochs "$1" --model "$2"
srun python training_geobench_so2sat.py --max_epochs "$1" --model "$2"