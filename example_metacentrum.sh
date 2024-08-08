#!/bin/bash
#PBS -N RunExperimentECNN
#PBS -q gpu
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1
#PBS -l walltime=24:00:00
#PBS -m ae

echo ${PBS_O_LOGNAME:?This script must be run under PBS scheduling system, execute: qsub $0}

# Define variables
HOMEDIR=/storage/brno2/home/$USER   # Home directory
BASE_DIR=$HOMEDIR/data              # Data directory
ETADL=$HOMEDIR/mqtt-edl             # Path to the repository
PYTHON_ENV=$ETADL/tf-venv           # Path to Python environment
source $PYTHON_ENV/bin/activate     # Activate Python environment
HOSTNAME=`hostname -f`              # Hostname
EXECMAIL=`which mail`               # Mail command
CONFIG_DIR=$ETADL/conf              # Configuration directory

# Send email
$EXECMAIL -s "[JP-METACENTRUM-JOB] Agent Sweep job is running on $HOSTNAME" $PBS_O_LOGNAME << EOFmail
Experiment has started.
Host domain $HOSTNAME
Experiment file $EXPERIMENT_FILE

Login to server:
ssh $USER@$HOSTNAME

To monitor standard output:
tail -f -n 50 /var/spool/pbs/spool/${SCRATCHDIR#/scratch.ssd/$USER/job_}.OU

Scratch dir:
$SCRATCHDIR
EOFmail

echo "Copy netrc credentials"
cp $HOMEDIR/.netrc ~/.netrc         # Copy netrc credentials to be sure that wandb will work

echo "Running experiment"
cd $ETADL
wandb agent <sweep_id>              # Run the experiment
