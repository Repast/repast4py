#!/bin/bash

#SBATCH --time=01:00:00
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=36
#SBATCH --mem=128G
#SBATCH --partition=dis
#SBATCH --account=CONDO
#SBATCH --output=/home/%u/sbatch_out/zombies.job.%j

module load gcc/8.2.0-xhxgy33
module load mvapich2/2.3-bebop-a66r4jf
module load anaconda3/5.2.0

export PYTHONPATH=/lcrc/project/EMEWS/bebop/repos/repast4py/src
cd /lcrc/project/EMEWS/bebop/repos/repast4py

srun -n 360 python ./src/zombies/zombies.py ./src/zombies/zombie_model.props "{\"stop.at\" : 100, \"human.count\" : 3000000, \"zombie.count\" : 6000, \"world.width\" : 1008, \"world.height\" : 1008}"

# MPICH
# module load gcc/8.2.0-g7hppkz
# module load mpich/3.3-verbs-ipsk4eg
# module load anaconda3/5.2.0

# export PYTHONPATH=/lcrc/project/EMEWS/bebop/repos/repast4py/src
# cd /lcrc/project/EMEWS/bebop/repos/repast4py

# mpiexec -n 360 python ./src/zombies/zombies.py ./src/zombies/zombie_model.props "{\"stop.at\" : 100, \"human.count\" : 3000000, \"zombie.count\" : 6000, \"world.width\" : 1008, \"world.height\" : 1008}"
