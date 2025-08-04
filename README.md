
# ONeSAMP_ML


ONeSAMP_ML computes the effective population size of genomic data sets.
This program takes a file in GENEPOP format and computes five summary statistics. 
The software then uses supervised machine learning to estimate the effective population size using the summary statistics as input features to the models.
ONeSAMP_ML enables effective population size inference using a range of machine learning models, including regularized regression techniques such as Lasso and Ridge regression, as well as ensemble methods like Random Forest and XGBoost.

It is strongly recommended that users read the accompanying manuscript before applying ONeSAMP_ML to their data. 



# Usage Overview
1. The program can be executed on MAC OS or LINUX system.

2. Python 3.8 or later is required to run the program

# Getting Started
1. Make a new ONeSAMP_ML directory

        mkdir ONeSAMP_ML
        cd ONeSAMP_ML
2. Clone the repository

        git clone [https://github.com/NiharaDeSilva/ONeSAMP_ML.git]
3. Give the Permission to the ONeSAMP file under the build directory

        chmod 777 build/OneSamp

# Running ONeSAMP_ML

usage: python main [--s number of trails] [--o input]
```
positional arguments:
   input  input file name

optional arguments:
    --n     Flag for Monomorphic loci (default: False)
    --m     Minimum Allele Frequency (size: 0-1)
    --r     Mutation Rate (size: 0-1)
    --lNe   Lower of Ne Range (size: 10-)
    --uNe   Upper of Ne Range (size: -500)
    --lT    Lower of Theta Range (size: 1-)
    --uT    Upper of Theta Range (size: -10)
    --s     Number of ONeSAMP Trials (size: 1000-50000)
    --lD    Lower of Duration Range (size: 2-)
    --uD    Upper of Duration Range (size: -8)
    --i     Missing data for individuals (size: 0-1)
    --l     Missing data for loci (size: 0-1)
```


Run the program

        python main --s 1000 --o exampleData/genePop10Ix30L > output.txt

# Getting Help

If you have any issues or anyquestions, please feel free to contact us at suhashidesilva@ufl.edu or through the GitHub Issues.






 
