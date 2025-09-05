# EColi-UTI-Predictions
Using AST data from UTI causing E. coli isolates to predict subpopulation membership.

The code in this repository trains a Random Forest classifier and an XGBoost classifier to predict whether or not a bacterial isolate belongs to ST131 clade C.  This prediction is made using disk diffusion measurements in milimetres from antibiotic susceptibility testing for ciprofloxacin, ceftazidim, and gentamicin.

As is mentioned in the corresponding manuscript (placeholder for DOI), this repository is meant to serve as an example. The models available here are not intended or expected to be generalizable to cases outside Norwegian *E. coli* ST131-C membership prediction.

## Repository Description
 - `/data` : Folder contianing all raw and pre-processed data sheets.  The AST data associtead with genomic UTI data shared by Handal, Kaspersen et al. is not shared in this repository.
 - `/scripts`: This folder contains all the code used to process data, train models on sequenced *E. coli* BSI isolates, and use trained models to make predictions on unsequenced BSI and UTI isolates.
    - `/scripts/main.py` : Main script for loading data, training and evaluating classifiers, and making predictions on unsequenced isolates.
    - `/scripts/dataproc.py` : File for preprocessing raw spreadsheets containing AST data.  Can be run independently but doesn't need to be.
    - `/scripts/classifierutils.py` : file containing all supporting functions for training and evaluating the classifiers used in this study.
    - `/scripts/test_genomic_UTI_samples.py` : python script used to test the predictive models on the genomic UTI data shared by [Handal et al.](https://doi.org/10.1093/jac/dkaf130) (raw data not included in this repository)
    - `/scripts/plot_ast_distributions.R` : script for generating boxplots and violin plots used in the manuscript.
    - `/scripts/test_genomic_UTI_samples.py` : script to test the models on the genomic UTI data shared by Handal, Kaspersen et al.
 - `/models`: All trained models are saved here.
 - `/output`: All output files and summaries from training and prediction are in this folder.

### Output files

Output files with the prefix `NORM-` correspond to the sequenced BSI isolates used for training and testing.  Prefixes `UTI-`, `BSI-`, and `UTI-BSI-` are model inferrences on unsequenced UTI and BSI data.

All .csv files included are the prediction outputs for all samples in each dataset.  All .txt files are summaries of each model's performance on the test and train data splits of the sequenced BSI datset.  The `yearly_fractions.png` images are visualisations of the yearly prevalence of ST131-C based on model predictions and (where available) genome sequence data. The included `correlations.png` files are calibration plots for predicted yearly prevalences of ST131-C compared to the prevalence observed in the sequenced isolates.

Two separate training methods were used in this study due to a re-standardisation of laboratory practices in collecting AST data.
1. `combined` : Train a model on all data from all available years. Use this model for inference of unsequenced isolates.
2. `split` : Train one model on data from 2010 and earlier, train a second model on data from 2011 onwards. Only compute inferrences with a model corresponding to the appropriate isolation year.

As a result, there are several model evaluation files.
- `NORM-combined` files correspond to method 1 above.
- `NORM-split` files correspond to method 2 above. These files (without `-old` or `-new`) are evaluations and predictions based on the combined inferences of models trained on both time periods.
    - `NORM-split-old` corresponds to evaluation of the model trained and evaluated on data from 2010 and earlier.
    - `NORM-split-new` corresponds to evaluation of the model trained and evaluated on data frmo 2011 and onwards.


## Reproducing
All results can be reproduced by running `main.py` in the `/scripts` folder.  The python packages needed to run this script are listed below:
- numpy
- scipy
- pandas
- scikit-learn
- xgboost
- altair
- vl-convert-python
- openpyxl

### Run with conda
A conda environment containing the necessary packages can be created using the following commands from this repository's directory.

    conda env create -f ecoli_ml_conda_env.yml

If the environment is successfully created and all packages are installed, the code can be run as follows:

    conda activate ecoli_ml_env
    python scripts/main.py

### Run with pip
A virtual environment containing the necessary packages can also be created with pip.

    python -m venv .venv
    source .venv/bin/activate
    pip install -r pip_requirements.txt

If everything installed successfuly, the main script can be run to reproduce the outputs.

    python scripts/main.py

### Run with docker
In case all else fails, a docker container is also available to run the code: (https://hub.docker.com/r/theodorross/ecoli-uti-predictions)
The line below run from this repository's main directory will run the training, evaluation, and prediction script.

    docker run --rm --mount type=bind,src=$(pwd),dst=$(pwd) -w $(pwd) -t theodorross/ecoli-uti-predictions scripts/main.py

