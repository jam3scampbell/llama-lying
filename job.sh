  sbatch <<EOF
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=localizing_lying_llama
#SBATCH --output=job_genacts_1.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --partition=single
python3 gen_acts_1.py
EOF

  sbatch <<EOF
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=localizing_lying_llama
#SBATCH --output=job_genacts_2.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --partition=single
python3 gen_acts_2.py
EOF