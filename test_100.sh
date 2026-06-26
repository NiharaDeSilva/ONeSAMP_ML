#!/bin/bash
#SBATCH --job-name=oneSampML    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=suhashidesilva@ufl.edu     # Where to send mail
#SBATCH --ntasks=1		      # Number of tasks
#SBATCH --cpus-per-task=8	      # Number of cores per task
#SBATCH --mem=40gb                     # Job memory request
#SBATCH --time=90:00:00               # Time limit hrs:min:sec
#SBATCH --output=log_files/serial_test_%j.log   # Standard output and error log
#SBATCH --account=boucher

# Increase file descriptor limit
ulimit -n 8192  # Set this to a higher number like 8192 or 16384
pwd; hostname; date

#module load R/4.1
module load conda
conda activate /blue/boucher/suhashidesilva/.conda/envs/my_env

echo "Python path:"
which python
python -c "import matplotlib; print('matplotlib OK')"


chmod +rwx /blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/build/OneSamp
#export PYTHONPATH=$PYTHONPATH:/blue/boucher/suhashidesilva/2025/WFsim/

echo "Running plot script on multiple CPU cores"
#output="/blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/output_tuning2"

#python /blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/main.py --s 20000 --o /blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/data_100/genePop100x1000_1 > /blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/output_tuning2/genePop100x1000.out


folder="/blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/data_100/samples/1000"
output="/blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/Final/output_ml_100"
MODE="infer"  # simulate | tune | train | infer


#Iterate through the files in the folder
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        filename_no_extension="${filename%.*}"
        output_file="$output/${filename_no_extension}"
        python /blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/main.py --mode "$MODE" --s 20000 --o "$file" > "$output_file"
        echo "Processed $file and saved output to $output_file"
    fi
done

date
