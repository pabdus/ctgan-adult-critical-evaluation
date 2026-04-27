# Data Directory

This directory contains the raw data used in the project. The dataset itself is **not included in the repository** to keep the repo lightweight and respect data licensing.

## Dataset: Adult Census Income (UCI Machine Learning Repository)

**Source:** Becker, B., & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5XW20

**Description:** Census data extracted from the 1994 U.S. Census database. 32,561 instances with 14 attributes plus a binary target indicating whether annual income exceeds $50K.

## How to obtain the data

### Option 1: Direct download from Kaggle

1. Download from: https://www.kaggle.com/datasets/uciml/adult-census-income
2. Save the file as `data/raw/adult.csv`

### Option 2: From UCI directly

Download `adult.data` and `adult.names` from:
https://archive.ics.uci.edu/ml/machine-learning-databases/adult/

Then convert to CSV format with appropriate column headers.

## Expected file structure after download