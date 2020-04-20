1. Navigate to https://www.kaggle.com/c/google-quest-challenge/data for the challenge page

2. Run `analysis.py` to generate all plots, which are placed in the plots/ directory

3. Run `preprocess_data.py`

4. Run `train.py`

Steps to generate labeled CSV using saved logistic regression models

1. Parse squad json into csv, run the command - `python parse_squad_to_csv.py`. This will generate train-v2.0.csv and dev-v2.0.csv in the squad_dataset folder.

2. cd into 'log_reg' directory - `cd log_reg`

3. Run the command - `python read_and_label_squad.py`

3. The labeled csv will be generated in the squad_dataset folder, dev-v2.0_labeled_log_reg_context.csv and train-v2.0_labeled_log_reg_context.csv.