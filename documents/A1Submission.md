operation: https://github.com/janvandermeulen/REMLA-group10 

## Comments for A1
### Task 1: Organise your training pipeline following machine learning project best practices.
Pull request: https://github.com/janvandermeulen/REMLA-group10/pull/1 and https://github.com/janvandermeulen/REMLA-group10/pull/2 
Contributors: Shayan Ramezani and Jan van der Meulen
Reviewers: Jan van der Meulen,  Shayan Ramezani, Michael Chan, and Remi Lejeune. 

We chose poetry to handle all the packages. Instructions to set-up the project are added in the README. The codebase was written such that DVC can do a step-by-ste- reproduction. 

### Task 2: Enable collaborative development through a pipeline management tool (DVC)
Pull request: https://github.com/janvandermeulen/REMLA-group10/pull/4  
Contributor: Remi Lejeune
Reviewer: Michael Chan

We uploaded the data to the a remote gdrive cloud bucket using dvc. The data is now versioned and can be accessed by all team members.
Furthermore, we created a reproduction pipeline. We have encountered some issues with DVC pull and it may not pull from cloud, in which case run dvc repro to reproduce the files.

### Task 3: Report metrics using DVC
Pull request: https://github.com/janvandermeulen/REMLA-group10/pull/4  
See description of previous task. 

### Task 4: Audit code quality
Pull request: https://github.com/janvandermeulen/REMLA-group10/pull/3
Contributor: Michael Chan
Reviewer: Jan van der Meulen, Remi Lejeune and Shayan Ramezani

We used pylint and bandit to audit the code quality. The README provides instructions on how to run these tools. We fixed all the errors that the tools showed. Explanation for some of the configuration settings for both pylint and bandit:

A regex was created to accept names with a single capital letter between "_" as those are common names for matrix variables in data science, example of accepted names by regex: X_train, raw_X_train and X.
TODO warnings have been suppressed temporarily. As this is still the first version there are still many things that could be improved that have been
tagged as TODO for now, this should not affect the code quality.
The number of arguments and local variables allowed has been increased as it is common in data science to separate data such as train and test in separate local variables, this results in relatively more variables used.
Bandit warning B106 about potential hardcoded access tokens has been suppressed as it falsely triggers on the usage of the word token which is prevalent in data science projects and has nothing to do with password/auth tokens.

