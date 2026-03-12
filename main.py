#!/usr/bin/python
import argparse
import os
import numpy as np
import pandas as pd
import time
import random
import multiprocessing
import concurrent.futures
import sys
import shutil
from statistics import statisticsClass
from sklearn.utils import resample
from models.train import run_model_training
from models.model_utils import run_all_models

import config as cfg
from models.tuning import train_and_tune_models

NUMBER_OF_STATISTICS = 5
t = 1
DEBUG = 0  ## BOUCHER: Change this to 1 for debuggin mode
# OUTPUTFILENAME = "priors.txt"


directory = "temp"

path = os.path.join("./", directory)

BASE_PATH = cfg.BASE_PATH
path = cfg.TEMP_DIR
output_path = cfg.OUTPUT_PATH
POPULATION_GENERATOR = cfg.POPULATION_GENERATOR


def getName(filename):
    (_, filename) = os.path.split(filename)
    return filename


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
parser.add_argument("--al", type=str, help="allpop stats file")
# parser.add_argument("--ip", type=str, help="inputpop stats file")

# parser.add_argument("--md", type=str, help="Model Name")

args = parser.parse_args()

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

upperNe = 400
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

allPopStats_path = "data_100/allPopStats_genePop100x1000"
if (args.al):
    allPopStats_path = str(args.al)

# if (args.ip):
#     inputStats_path = str(args.ip)

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

cfg.config.numLoci = numLoci
cfg.config.sampleSize = sampleSize

##Creating input file & List with intial statistics
textList = [str(inputFileStatistics.stat1_new), str(inputFileStatistics.stat2), str(inputFileStatistics.stat3),
             str(inputFileStatistics.stat4), str(inputFileStatistics.stat5)]
inputStatsList = textList

'''
inputPopStats = output_path + "inputPopStats_" + getName(fileName)
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

print("starting population simulations:")

# Generate random populations and calculate summary statistics
def processRandomPopulation(x):
    loci = inputFileStatistics.numLoci
    sampleSize = inputFileStatistics.sampleSize
    proc = multiprocessing.Process()
    process_id = os.getpid()
    # change the intermediate file name by process id
    intermediateFilename = str(process_id) + "_intermediate_" + getName(fileName) + "_" + str(t)
    intermediateFile = os.path.join(directory, intermediateFilename)
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


try:
    os.mkdir(path)
except FileExistsError:
    pass


def main():
    # Parallel process the random populations and add to a list
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        for result in executor.map(processRandomPopulation, range(numOneSampTrials)):
            try:
                results_list.append(result)
            except Exception as e:
                print(f"Generated an exception: {e}")


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    main()


try:
    shutil.rmtree(directory)
except FileNotFoundError:
    print(f"Directory '{directory}' not found.")

allPopStats = output_path + "allPopStats_" + getName(fileName)
with open(allPopStats, 'w') as file:
    for result in results_list:
        file.write('\t'.join(map(str, result)) + '\n')

simulation_time = time.time()

print("-----Population simulation time %s seconds -----" % (time.time() - start_time))



########################################
# FINISHING ALL POPULATIONS
########################################

# Assign input and all population stats to dataframes with column names
allPopStatistics = pd.DataFrame(results_list, columns=['Ne','Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt'])
inputStatsList = pd.DataFrame([textList], columns=['Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance', 'Fix_index', 'Emean_exhyt'])

'''


# =========================================================
# TUNE MODELS
# =========================================================

if __name__ == "__main__":
    input_sample_size ="f{sampleSize}x{numLoci}"
    allPopStatistics = pd.read_csv(allPopStats_path, sep='\t', header=None, names=[ 'Ne','Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance','Fix_index','Emean_exhyt'])
    results = train_and_tune_models(
        allPopStatistics=allPopStatistics,
        input_text_list=textList,
        input_sample_size=input_sample_size,
        output_dir=output_path,
        n_splits=5,
        random_state=42)

    print(results["results_df"])

# =========================================================
# TRAIN MODEL
# =========================================================

# inputStatsList = pd.DataFrame([textList], columns=['Gametic_equilibrium','Mlocus_homozegosity_mean','Mlocus_homozegosity_variance','Fix_index','Emean_exhyt'])
# allPopStatistics = pd.read_csv(allPopStats_path, sep='\t', header=None, names=[ 'Ne', 'Gametic_equilibrium', 'Mlocus_homozegosity_mean', 'Mlocus_homozegosity_variance','Fix_index','Emean_exhyt'])
# run_model_training('all', allPopStatistics, inputStatsList, numLoci, sampleSize)

# =========================================================
# INFERENCE
# =========================================================

# train_path  = os.path.join(output_path, f'allPopStats_genePop{sampleSize}x{numLoci}_1')
# run_all_models(sampleSize, numLoci, inputStatsList, train_path)
