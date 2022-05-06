# UW-Madison GI Tract Image Segmentation

This repository contains files and code related to our submission to the [Tract Image Segmentation competition on Kaggle](https://www.kaggle.com/competitions/uw-madison-gi-tract-image-segmentation).
We picked this challenge as a project for the TU Delft course Seminar Computer Vision by Deep Learning.

## Setting up

- Create an account on Kaggle and enroll in the competition.
- Install and authenticate with Kaggle API, please see the [Kaggle API Getting Started](https://www.kaggle.com/docs/api#getting-started-installation-&-authentication) documentation.

## Running code on Kaggle

- Run the included Python script: `python push_to_kaggle.py`
- **Important**: Only code in the `scv_notebook.ipynb` notebook is automatically executed on Kaggle. If some cell errors, all cells below are not executed. Utility functions can (only) be defined in the `scv_utility.py` script and can be imported in the notebook.
- Tests are not executed on Kaggle.

## Running tests locally

- Tests can be written in the `scv_tests.py` file. Multiple test classes can be created. **Important**: Only methods that start with `test_` are considered a test.
- Run the test suite with `python scv_tests.py`.
