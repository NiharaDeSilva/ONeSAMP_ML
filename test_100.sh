#!/bin/bash
# Released under the GNU GPLv3; see LICENSE for details.
# Developed by Boucher Lab,
#SBATCH --job-name=oneSampML    # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=suhashidesilva@ufl.edu     # Where to send mail
#SBATCH --ntasks=1		      # Number of tasks
#SBATCH --cpus-per-task=16	      # Number of cores per task
#SBATCH --mem=40gb                     # Job memory request
#SBATCH --time=90:00:00               # Time limit hrs:min:sec
#SBATCH --output=/blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/log_files/serial_test_%j.log   # Standard output and error log
#SBATCH --account=boucher

# Increase file descriptor limit
ulimit -n 8192  # Set this to a higher number like 8192 or 16384
pwd; hostname; date

PROJECT_DIR="/blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML"
cd "$PROJECT_DIR" || exit 1
mkdir -p log_files

#module load R/4.1
module load conda
conda activate /blue/boucher/suhashidesilva/.conda/envs/my_env

echo "Python path:"
which python
python -c "import matplotlib; print('matplotlib OK')"


chmod +rwx "$PROJECT_DIR/build/OneSamp"
#export PYTHONPATH=$PYTHONPATH:/blue/boucher/suhashidesilva/2025/WFsim/

echo "Running plot script on multiple CPU cores"
#output="/blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/output_tuning2"

#python /blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/main.py --s 20000 --o /blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/data_100/genePop100x1000_1 > /blue/boucher/suhashidesilva/2025/Revision/ONeSAMP_ML/output_tuning2/genePop100x1000.out


folder="$PROJECT_DIR/data_100/samples/1000"
output="$PROJECT_DIR/Final/output_ml_100"
memory_output="$PROJECT_DIR/Final/memory_reports"
MODE="infer"      # train | infer
MODELS="all"      # all | rf | xb | ls | rd | comma-separated, e.g. rf,xb
ALLPOPSTATS=""    # Optional: set a fixed allPopStats path, or leave empty to use the default path from main.py

mkdir -p "$output"
mkdir -p "$memory_output"

#Iterate through the files in the folder
for file in "$folder"/*; do
    if [ -f "$file" ]; then
        filename=$(basename -- "$file")
        filename_no_extension="${filename%.*}"
        output_file="$output/${filename_no_extension}"
        memory_file="$memory_output/${filename_no_extension}.memory.txt"
        cmd=(python "$PROJECT_DIR/main.py" --mode "$MODE" --models "$MODELS" --s 20000 --o "$file")
        if [ -n "$ALLPOPSTATS" ]; then
            cmd+=(--allpopstats "$ALLPOPSTATS")
        fi
        /usr/bin/time -v "${cmd[@]}" > "$output_file" 2> "$memory_file"
        echo "Processed $file and saved output to $output_file"
        echo "Memory report saved to $memory_file"
    fi
done

date
