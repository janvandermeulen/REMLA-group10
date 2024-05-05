# REMLA project for group10
In this assignment we will be transferring a small kaggle model to a professional development environment. We will be using the following tools:
- dvc for data version control and machine learning reproducibility.
- git for version control.
- poetry for dependency management.

### How to run
To run this codee you need to have poetry installed. You can install the packages by running the following commands:
- ```poetry install```
- ```poetry shell```
To retrieve the data and run the pipeline:
- ```dvc pull```
- ```dvc repro```
To run the code quality metrics:
- ```pylint ./phishing_detection```
- ```bandit ./```