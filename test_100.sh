#!/bin/bash
#SBATCH --job-name=oneSamp    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=suhashidesilva@ufl.edu     # Where to send mail
#SBATCH --ntasks=1		      # Number of tasks
#SBATCH --cpus-per-task=16	      # Number of cores per task
#SBATCH --mem=80gb                     # Job memory request
#SBATCH --time=240:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log

# Increase file descriptor limit
ulimit -n 8192  # Set this to a higher number like 8192 or 16384
pwd; hostname; date

#module load R/4.1
module load conda
source activate /blue/boucher/suhashidesilva/conda_envs/my_env

chmod +rwx /blue/boucher/suhashidesilva/2025/ONeSAMP_ML/build/OneSamp
#export PYTHONPATH=$PYTHONPATH:/blue/boucher/suhashidesilva/2025/WFsim/

echo "Running plot script on multiple CPU cores"

#python /blue/boucher/suhashidesilva/2025/ONeSAMP_ML/main.py --s 10000 --o /blue/boucher/suhashidesilva/2025/ONeSAMP_ML/data_70/genePop5Ix5L > /blue/boucher/suhashidesilva/2025/ONeSAMP_ML/genePop5Ix5L.out

folder="/blue/boucher/suhashidesilva/2025/ONeSAMP_ML/data_70/samples"
output="/blue/boucher/suhashidesilva/2025/ONeSAMP_ML/output/"


#Iterate through the files in the folder
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        filename_no_extension="${filename%.*}"
        output_file="$output/${filename_no_extension}"
        python /blue/boucher/suhashidesilva/2025/ONeSAMP_ML/main.py --s 20 --o "$file" > "$output_file"
        echo "Processed $file and saved output to $output_file"
    fi
done

date
