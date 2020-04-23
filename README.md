Navigate to https://www.kaggle.com/c/google-quest-challenge/data for the challenge page

## LSTM (QUEST labeling model)

1. Run `cd lstm` to navigate to the LSTM folder

2. Run `pip3 install -r requirements.txt`

3. Run `python3 preprocess_data.py` to preprocess the QUEST dataset

4. Run `python3 train.py`

5. Run `python3 test.py` to generate a test submission file for Kaggle

6. The final, best model will be saved in the models/ folder

## Logistic Regression (QUEST labeling model)

Steps to obtain the training accuracy (test accuracy is displayed on Kaggle) using Spearman score -

1. cd into 'log_reg' directory - `cd log_reg`

2. We then run logistic regression to generate the features, train and save the model - `python logistic_regression.py` (Will take around 5 minutes)

3. Above command will display the Spearman score for each label and also the overall average Spearman score.

## Naive Bayes (QUEST labeling model)

## Bi-directional Attention Flow (BiDAF) (QA baseline model)

The initial code repository is located at https://github.com/ElizaLo/Question-Answering-based-on-SQuAD .

1. Run `cd qa` to navigate to the qa folder

2. Run `pip3 install -r requirements.txt`

3. Run `pip3 install -r requirements.txt` to install all Python3 libraries

4. Run `python3 -m spacy download en` to download necessary Spacy models. These should be located in your `Python/Python36/Lib/site-packages/en_core_web_sm/` folder.

5. Modify data_dir, spicy_en, and glove in `config.py` to be the locations where you want to download SQuAD to, the location from step 3 of your Spacy models on your system, and the location of your pre-trained GloVe embeddings downloaded from https://nlp.stanford.edu/projects/glove/ and extracted to that folder.

6. Run `python3 make_dataset.py` to preprocess the SQuAD dataset and create the vocabulary.

7. Run `python3 train.py` to train and save the BiDAF model.

8. Run `python3 test.py` to test the BiDAF model.

## Steps to show dataset analysis

Before running the below programs, please download the SQuAD1.1 dataset from https://www.wolframcloud.com/objects/d91733a5-57f5-40fe-8e09-2f5285d21fe6, create a `data` directory, and place the file into that directory. Ensure the file is titled SQuAD-v1.1.csv before continuing.

1. Run `analysis.py` to generate all plots of the QUEST datasets' features, which are placed in the plots/ directory

2. Run `quest_analysis.py` to generate QUEST dataset statistics.

3. Run `squad_analysis.py` to generate SQuAD dataset statistics.

## Steps to generate error analysis plots (Plots already present in `plots_error_analysis` folder, run steps if not present)

1. cd into 'log_reg' directory - `cd log_reg`

2. Run logistic regression if not done above - `python logistic_regression.py`

3. Now, run the command - `python error_analysis.py` to generate the error analysis plots in the `plots_error_analysis` folder.

## Steps to generate transformed SQuAD using saved logistic regression models

1. First we parse squad json into csv for a more readable format, run the command from the parent directory- `python parse_squad_to_csv.py`. This will generate train-v1.1.csv and dev-v1.1.csv in the squad_dataset folder.

2. cd into 'log_reg' directory - `cd log_reg`

3. Run the command - `python read_and_label_squad.py`. This will use the saved logistic regression models to label squad. This will take ~5 minutes.

3. The labeled csv will be generated in the `squad_labelled` folder, dev-v1.1_labeled.csv and train-v1.1_labeled.csv.

## Steps to generate plots that analyze the transformed SQuAD data

1. cd into 'log_reg' directory - `cd log_reg`

2. Run the command - `python analysis_labeled.py`

3. The plots will be generated in the `plots_labeled` folder.
