  sbatch <<EOF
#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name=localizing_lying_llama
#SBATCH --output=job_genacts_1_1gpu.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --partition=single
python3 patching.py
EOF

#   sbatch <<EOF
# #!/bin/bash
# #SBATCH --time=24:00:00
# #SBATCH --job-name=localizing_lying_llama
# #SBATCH --output=job_genacts_2_1gpu.txt
# #SBATCH --nodes=1
# #SBATCH --gpus-per-node=1
# #SBATCH --partition=single
# python3 gen_acts_2.py
# EOF