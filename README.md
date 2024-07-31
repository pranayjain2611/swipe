## Cognitive State Prediction with ML: fMRI Hurst Exponent Analysis


# FMRI Analysis Project

This project contains a series of Jupyter notebooks and Python scripts for analyzing fMRI data. The notebooks cover different aspects of the analysis, including hurst exponent calculation, spectral analysis, connectivity analysis, and evaluation using SVM and LDA.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
  - [1-fmri_hurst.ipynb](#1-fmri_hurstipynb)
  - [2-spectral_analysis.ipynb](#2-spectral_analysisipynb)
  - [3-connectivity_analysis.ipynb](#3-connectivity_analysisipynb)
  - [4-svm_evaluation.ipynb](#4-svm_evaluationipynb)
  - [5-lda_evaluation.ipynb](#5-lda_evaluationipynb)
- [Scripts](#scripts)

## Installation

To set up the environment for this project, you can use either the `requirements.txt` file or the `environment.yml` file.

### Using `requirements.txt`

1. Create a new conda environment:
    ```bash
    conda create --name nilearn python=3.8
    ```
2. Activate the environment:
    ```bash
    conda activate nilearn
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Using `environment.yml`

1. Create the environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
2. Activate the environment:
    ```bash
    conda activate nilearn
    ```

## Usage

The project includes several Jupyter notebooks, each designed for a specific aspect of the fMRI data analysis.

### 1-fmri_hurst.ipynb

This notebook calculates the Hurst exponent for the fMRI data using DFA. The Hurst exponent is used to measure the long-term memory of time series data.

### 2-spectral_analysis.ipynb

This notebook performs spectral analysis on the fMRI data to identify different frequency components.

### 3-connectivity_analysis.ipynb

This notebook analyzes the functional connectivity of the brain regions based on the fMRI data. Connectivity analysis helps in understanding the interactions between different brain regions.

### 4-svm_evaluation.ipynb

This notebook uses Support Vector Machine (SVM) for evaluating the classification performance on the fMRI data. SVM is a powerful machine learning technique used for classification tasks.

### 5-lda_evaluation.ipynb

This notebook uses Linear Discriminant Analysis (LDA) for evaluating the classification performance on the fMRI data. LDA is a statistical method used for finding the linear combinations of features that best separate different classes.

## Scripts

The project also includes supporting python scripts for the notebooks:

- `anova_test.py`: Performs ANOVA tests on the data.
- `data_loader.py`: Contains functions for loading the fMRI data.
- `hurst_dfa.py`: Implements the Detrended Fluctuation Analysis (DFA) method for calculating the Hurst exponent.
- `model.py`: Contains machine learning models for analysis.
- `postprocessing.py`: Contains functions for postprocessing the analysis results.
- `t_test.py`: Performs t-tests on the data.
