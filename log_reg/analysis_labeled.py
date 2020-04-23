import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("./squad_labelled/train-v1.1_labeled.csv")

# create "plots_labeled" folder
if not os.path.exists("plots_labeled"):
    os.mkdir("plots_labeled")

# data analysis on the 30 attributes
attribute_column_names = list(data.columns[6:])  # only get the column names belonging to the 30 attributes
print("Number of attribute column names: {}".format(len(attribute_column_names)))
print("Attribute column names: \n{}".format(attribute_column_names))
for column_name in attribute_column_names:
    column_data = data[column_name]  # extract this column's data

    print("{} : {}".format(column_name,
                           column_data.describe()))  # get the 5 number summary of this column/attribute's values

    # create a histogram with 10 bins for this attribute and save the plot in the plots_labeled/ directory
    plt.hist(column_data, bins=10)
    plt.title("{} histogram".format(column_name))
    plt.xlabel("Attribute value")
    plt.ylabel("Counts")
    save_directory = os.path.join("plots_labeled", "{}_histogram.jpg".format(column_name))
    plt.savefig(save_directory)
    plt.figure()
