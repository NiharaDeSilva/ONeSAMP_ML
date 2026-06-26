#!/usr/bin/python
import argparse
import os
import sys
import pandas as pd
import time
import random
import multiprocessing
import concurrent.futures
import shutil
from statistics import statisticsClass
import models.train as train
import models.model_utils as model_utils

from config import configClass, OUTPUT_PATH, BASE_PATH, POPULATION_GENERATOR, TEMP_DIR

NUMBER_OF_STATISTICS = 5
t = 1
DEBUG = 0  ## BOUCHER: Change this to 1 for debuggin mode
# OUTPUTFILENAME = "priors.txt"


temp_dir = TEMP_DIR
output_path = OUTPUT_PATH


def getName(filename):
    (_, filename) = os.path.split(filename)
    return filename


MODEL_ALIASES = {
    "rf": "RandomForest",
    "xb": "XGBoost",
    "ls": "Lasso",
    "rd": "Ridge",
}


def parse_model_codes(model_codes):
    if model_codes is None or model_codes.lower() == "all":
        return None

    selected_models = []
    invalid_codes = []
    for code in model_codes.split(","):
        normalized_code = code.strip().lower()
        if not normalized_code:
            continue
        model_name = MODEL_ALIASES.get(normalized_code)
        if model_name is None:
            invalid_codes.append(code.strip())
            continue
        if model_name not in selected_models:
            selected_models.append(model_name)

    if invalid_codes:
        valid_codes = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unknown model code(s): {', '.join(invalid_codes)}. Valid codes: {valid_codes}, or all.")

    if not selected_models:
        raise ValueError("No valid model codes were provided.")

    return selected_models


#############################################################
start_time = time.time()

parser = argparse.ArgumentParser()
parser.add_argument("--m", type=float, help="Minimum Allele Frequency")
parser.add_argument("--r", type=float, help="Mutation Rate")
parser.add_argument("--lNe", type=int, help="Lower of Ne Range")
parser.add_argument("--uNe", type=int, help="Upper of Ne Range")
parser.add_argument("--lT", type=float, help="Lower of Theta Range")
parser.add_argument("--uT", type=float, help="Upper of Theta Range")
parser.add_argument("--s", type=int, help="Number of OneSamp Trials")
parser.add_argument("--lD", type=float, help="Lower of Duration Range")
parser.add_argument("--uD", type=float, help="Upper of Duration Range")
parser.add_argument("--i", type=float, help="Missing data for individuals")
parser.add_argument("--l", type=float, help="Missing data for loci")
parser.add_argument("--o", type=str, help="The File Name")
parser.add_argument("--t", type=int, help="Repeat times")
parser.add_argument("--n", type=bool, help="whether to filter the monomorphic loci", default=False)
parser.add_argument("--mode", type=str, choices=["train", "infer"], default="train",
                    help="Execution mode: train models or run inference")
parser.add_argument("--allpopstats", type=str,
                    help="Path to an existing allPopStats file for train/infer modes")
parser.add_argument("--models", type=str, default="all",
                    help="Comma-separated ML model codes for train/infer. Use rf, xb, ls, rd, or all. Default: all")

# parser.add_argument("--md", type=str, help="Model Name")

args = parser.parse_args()
try:
    selected_models = parse_model_codes(args.models)
except ValueError as exc:
    parser.error(str(exc))

#########################################
# INITIALIZING PARAMETERS
#########################################
#if (args.t):
#    t = int(args.t)

minAlleleFreq = 0.05
if (args.m):
    minAlleleFreq = float(args.m)

mutationRate = 0.000000012

if (args.r):
    mutationRate = float(args.r)

lowerNe = 4
if (args.lNe):
    lowerNe = int(args.lNe)

upperNe = 200
if (args.uNe):
    upperNe = int(args.uNe)

if (int(lowerNe) > int(upperNe)):
    print("ERROR:main:lowerNe > upperNe. Fatal Error")
    exit()

if (int(lowerNe) < 1):
    print("ERROR:main:lowerNe must be a positive value. Fatal Error")
    exit()

if (int(upperNe) < 1):
    print("ERROR:main:upperNe must be a positive value. Fatal Error")
    exit()

rangeNe = "%d,%d" % (lowerNe, upperNe)

lowerTheta = 0.000048
if (args.lT):
    lowerTheta = float(args.lT)

upperTheta = 0.0048
if (args.uT):
    upperTheta = float(args.uT)

rangeTheta = "%f,%f" % (lowerTheta, upperTheta)

numOneSampTrials = 20000
if (args.s):
    numOneSampTrials = int(args.s)

lowerDuration = 2
if (args.lD):
    lowerDuration = float(args.lD)

upperDuration = 8
if (args.uD):
    upperDuration = float(args.uD)

indivMissing = .2
if (args.i):
    indivMissing = float(args.i)

lociMissing = .2
if (args.l):
    lociMissing = float(args.l)

rangeDuration = "%f,%f" % (lowerDuration, upperDuration)

fileName = "data_100/genePop100x1000_1"

if (args.o):
    fileName = str(args.o)
else:
    print("WARNING:main: No filename provided.  Using oneSampIn")

if (DEBUG):
    print("Start calculation of statistics for input population")

rangeTheta = "%f,%f" % (lowerTheta, upperTheta)



#########################################
# STARTING INITIAL POPULATION
#########################################

inputFileStatistics = statisticsClass()

inputFileStatistics.readData(fileName)
inputFileStatistics.filterIndividuals(indivMissing)
inputFileStatistics.filterLoci(lociMissing)
if (args.n):
    inputFileStatistics.filterMonomorphicLoci()

#inputFileStatistics.test_stat1()
inputFileStatistics.test_stat1_new()
inputFileStatistics.test_stat2()
inputFileStatistics.test_stat3()
inputFileStatistics.test_stat5()
inputFileStatistics.test_stat4()

numLoci = inputFileStatistics.numLoci
sampleSize = inputFileStatistics.sampleSize

cfg = configClass()
cfg.numLoci = numLoci
cfg.sampleSize = sampleSize
train.set_size(cfg)

##Creating input file & List with intial statistics
textList = [str(inputFileStatistics.stat1_new), str(inputFileStatistics.stat2), str(inputFileStatistics.stat3),
             str(inputFileStatistics.stat4), str(inputFileStatistics.stat5)]
inputStatsList = textList

inputPopStats = os.path.join(BASE_PATH, f"inputPopStats_{getName(fileName)}")
with open(inputPopStats, 'w') as fileINPUT:
    fileINPUT.write('\t'.join(textList[0:]) + '\t')
fileINPUT.close()


if (DEBUG):
    print("Finish calculation of statistics for input population")

#############################################
# FINISH STATS FOR INITIAL INPUT  POPULATION
############################################

#########################################
# STARTING ALL POPULATIONS
#########################################

#Result queue
results_list = []

if (DEBUG):
    print("Start calculation of statistics for ALL populations")

#statistics1 = []
statistics1_new = []
statistics2 = []
statistics3 = []
statistics4 = []
statistics5 = []

#statistics1 = [0 for x in range(numOneSampTrials)]
statistics1_new = [0 for x in range(numOneSampTrials)]
statistics2 = [0 for x in range(numOneSampTrials)]
statistics3 = [0 for x in range(numOneSampTrials)]
statistics5 = [0 for x in range(numOneSampTrials)]
statistics4 = [0 for x in range(numOneSampTrials)]


# Generate random populations and calculate summary statistics
def processRandomPopulation(x):
    loci = inputFileStatistics.numLoci
    sampleSize = inputFileStatistics.sampleSize
    proc = multiprocessing.Process()
    process_id = os.getpid()
    # change the intermediate file name by process id
    intermediateFilename = str(process_id) + "_intermediate_" + getName(fileName) + "_" + str(t)
    intermediateFile = os.path.join(temp_dir, intermediateFilename)
    Ne_left = lowerNe
    Ne_right = upperNe
    if Ne_left % 2 != 0:
        Ne_left += 1
    num_evens = (Ne_right - Ne_left) // 2 + 1
    random_index = random.randint(0, num_evens - 1)
    target_Ne = Ne_left + random_index * 2
    target_Ne = f"{target_Ne:05d}"
    cmd = "%s -u%.9f -v%s -rC -l%d -i%d -d%s -s -t1 -b%s -f%f -o1 -p > %s" % (POPULATION_GENERATOR, mutationRate, rangeTheta, loci, sampleSize, rangeDuration, target_Ne, minAlleleFreq, intermediateFile)
    #print(minAlleleFreq, mutationRate,  lowerNe, upperNe, lowerTheta, upperTheta, lowerDuration, upperDuration, loci, sampleSize, intermediateFile)
    #run_simulation(minAlleleFreq, mutationRate,  lowerNe, upperNe, lowerTheta, upperTheta, lowerDuration, upperDuration, loci, sampleSize, intermediateFile)

    if (DEBUG):
        print(cmd)

    returned_value = os.system(cmd)

    if returned_value:
        print("ERROR:main:Refactor did not run")


    refactorFileStatistics = statisticsClass()

    refactorFileStatistics.readData(intermediateFile)
    refactorFileStatistics.filterIndividuals(indivMissing)
    refactorFileStatistics.filterLoci(lociMissing)
    #refactorFileStatistics.test_stat1()
    refactorFileStatistics.test_stat1_new()
    refactorFileStatistics.test_stat2()
    refactorFileStatistics.test_stat3()
    refactorFileStatistics.test_stat5()
    refactorFileStatistics.test_stat4()

    #statistics1[x] = refactorFileStatistics.stat1
    statistics1_new[x] = refactorFileStatistics.stat1_new
    statistics2[x] = refactorFileStatistics.stat2
    statistics3[x] = refactorFileStatistics.stat3
    statistics5[x] = refactorFileStatistics.stat5
    statistics4[x] = refactorFileStatistics.stat4


    # Making file with stats from all populations
    textList = [str(refactorFileStatistics.NE_VALUE), str(refactorFileStatistics.stat1_new),
                str(refactorFileStatistics.stat2),
                str(refactorFileStatistics.stat3),
                str(refactorFileStatistics.stat4), str(refactorFileStatistics.stat5)]

    return textList


os.makedirs(temp_dir, exist_ok=True)

def generate_all_pop_results():
    results_list = []
    ctx = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=16, mp_context=ctx) as executor:
        for result in executor.map(processRandomPopulation, range(numOneSampTrials)):
            results_list.append(result)

    return results_list


if __name__ == "__main__":
    start_time = time.time()

    allPopStats_path = args.allpopstats or os.path.join(OUTPUT_PATH, f"allPopStats_genePop{sampleSize}x{numLoci}_1")

    # Generate all population statistics when needed
    def write_all_pop_stats(results, path):
        with open(path, "w") as file:
            for result in results:
                file.write("\t".join(map(str, result)) + "\n")
        print(f"Wrote all population stats to {path}")

    def load_all_pop_statistics(path):
        return pd.read_csv(
            path,
            sep='\t',
            header=None,
            names=['Ne','Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt']
        )

    def ensure_all_pop_stats(path):
        if os.path.exists(path):
            print(f"Using existing all population stats: {path}")
            return None
        results = generate_all_pop_results()
        write_all_pop_stats(results, path)
        return results

    if args.mode == "train":
        ensure_all_pop_stats(allPopStats_path)
    elif args.mode == "infer":
        if not os.path.exists(allPopStats_path):
            print(f"ERROR: Inference mode requires existing allPopStats at {allPopStats_path}")
            sys.exit(1)
    else:
        print(f"ERROR: Unknown mode '{args.mode}'")
        sys.exit(1)

    inputStatsList = pd.DataFrame([textList], columns=['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt'])

    if args.mode == "train":
        allPopStatistics = load_all_pop_statistics(allPopStats_path)
        train.run_model_training(cfg, selected_models, allPopStatistics, inputStatsList)
        sys.exit(0)

    if args.mode == "infer":
        train_path = allPopStats_path
        inference_start = time.time()
        model_utils.run_all_models(cfg, inputStatsList, train_path, selected_models)
        inference_elapsed = time.time() - inference_start
        print(f"Inference time: {inference_elapsed:.2f} seconds")
        sys.exit(0)

    # Should never get here
    print(f"Completed mode: {args.mode}")
    sys.exit(0)
