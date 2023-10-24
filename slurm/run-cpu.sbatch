#!/bin/bash
#SBATCH --output="/projects/ec12/jinyueg/eprior-RAVE/slurm/log/run-cpu2.log"
#SBATCH --job-name="eprior"
#SBATCH --time=12:00:00     # walltime
#SBATCH --mem-per-cpu=8G   # memory per CPU core
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=ec12

module purge
module load Miniconda3/22.11.1-1
source ${EBROOTMINICONDA3}/etc/profile.d/conda.sh
conda deactivate &>/dev/null
conda init bash
conda activate /projects/ec12/jinyueg/conda/envs/eprior

cd /projects/ec12/jinyueg/eprior-RAVE/eprior/scripts

python run.py --config_path ../eprior/configs/VCTK_novelty.gin
