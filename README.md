1. Navigate to https://www.kaggle.com/c/google-quest-challenge/data for the challenge page

2. Run `analysis.py` to generate all plots, which are placed in the plots/ directory

3. Run `preprocess_data.py`

4. Run `train.py`

Logistic Regression

Steps to obtain the training accuracy (test accuracy is displayed on Kaggle) using Spearman score-

1. cd into 'log_reg' directory - `cd log_reg`

2. We then run logistic regression to generate the features, train and save the model - `python logistic_regression.py` (Will take around 5 minutes)

3. Above command will display the Spearman score for each label and also the overall average Spearman score.

Steps to show error analysis

1. cd into 'log_reg' directory - `cd log_reg`

2. Run logistic regression if not done above - `python logistic_regression.py`

3. Now, run the command - `python error_analysis.py` to generate the error analysis plots in the `plots_error_analysis` folder.

Steps to generate labeled CSV using saved logistic regression models

1. First we parse squad json into csv for a more readable format, run the command from the parent directory- `python parse_squad_to_csv.py`. This will generate train-v1.1.csv and dev-v1.1.csv in the squad_dataset folder.

2. cd into 'log_reg' directory - `cd log_reg`

3. Run the command - `python read_and_label_squad.py`. This will use the saved logistic regression models to label squad.

3. The labeled csv will be generated in the `squad_labelled` folder, dev-v1.1_labeled.csv and train-v1.1_labeled.csv.

Steps to generate plots that analyze the labelled squad data

1. cd into 'log_reg' directory - `cd log_reg`

2. Run the command - `python analysis_labeled.py`

3. The plots will be generated in the `plots_labeled` folder.
