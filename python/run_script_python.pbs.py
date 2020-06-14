#PBS -N romanovi_rotace
#PBS -S /bin/bash
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -l mem=32gb
#PBS -l walltime=100:00:00
cd $PBS_O_WORKDIR/

source /home/user_pool_2/vicar/miniconda3/bin/activate
conda activate pytorch

python main.py