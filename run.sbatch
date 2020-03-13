#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --time=0:30:00
#SBATCH --job-name="nlu_hw4"

module purge
module load cuda/10.0.130
module load gcc/6.3.0

# Replace with your NetID
NETID=js11133
source activate /home/${NETID}/.conda/envs/env

cd /scratch/${NETID}/nlu/code/transformers
export PYTHONPATH=/scratch/${NETID}/nlu/code/transformers/src:$PYTHONPATH
python /scratch/js11133/nlu/code/transformers/examples/run_glue.py \
	--data_dir /scratch/${NETID}/nlu/BoolQ \
	--model_type roberta \
	--task_name boolq \
	--do_train \
	--do_eval \
        --overwrite_output_dir \
    	--learning_rate 2e-5 \
   	--model_name_or_path roberta-base \
	--output_dir /scratch/${NETID}/nlu/out/ \
	--seed 5
