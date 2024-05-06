# REMLA project for group10
In this assignment we will be transferring a small kaggle model to a professional development environment. We will be using the following tools:
- dvc for data version control and machine learning reproducibility.
- git for version control.
- poetry for dependency management.

### How to run
To run this code you need to have poetry installed. 
You can install the packages by running the following commands:
(this should be executed in the phishing-detection folder)

- if the lock file is out of date:
- ```poetry lock --no-update```

- ```poetry install```
- ```poetry shell```

To retrieve the data and run the pipeline:
(this should be executed in the remla-group10 folder)
- ```dvc fetch```
- ```dvc pull``` (may not work)
- ```dvc repro```

To run the code quality metrics:
(this should be executed in the phishing-detection folder)
- ```pylint ./phishing_detection```
- ```bandit ./ -r```

The project will be restructured in the future such that there is a single root folder from which all scripts can be executed from.
