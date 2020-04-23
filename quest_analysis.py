import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")

# create "plots" folder
if not os.path.exists("plots"):
    os.mkdir("plots")

print("Extracted columns: \n{}".format(list(data.columns)))
question_titles = data["question_title"]
answers = data["answer"]
question_categories = data["category"]

print("Number of samples: {}".format(len(answers)))

# question analysis
question_avg_length = 0
for question in question_titles:
    question_avg_length += len(question.split(" "))
question_avg_length /= len(question_titles)
print("Average question length: {}".format(question_avg_length))

# answer analysis
answer_avg_length = 0
for answer in answers:
    answer_avg_length += len(answer.split(" "))
answer_avg_length /= len(answers)
print("Average answer length: {}".format(answer_avg_length))

# data analysis on the question categories
category_percentages = question_categories.value_counts(normalize = True, sort = True, ascending = False) # for each category, calculate which % of samples are in this category -- normalize = True turns the counts into percentages
categories = category_percentages.index.tolist()
percentages = category_percentages.values.tolist()
plt.pie(percentages, labels = categories, autopct = "%1.1f%%") # create pie chart
plt.title("Category distribution")
save_directory = os.path.join("plots", "categories_piechart.jpg")
plt.savefig(save_directory)
plt.figure()

# data analysis on the 30 attributes
attribute_column_names = list(data.columns[11:]) # only get the column names belonging to the 30 attributes
print("Number of attribute column names: {}".format(len(attribute_column_names)))
print("Attribute column names: \n{}".format(attribute_column_names))
for column_name in attribute_column_names:
    column_data = data[column_name] # extract this column's data

    print("{} : {}".format(column_name, column_data.describe())) # get the 5 number summary of this column/attribute's values

    # create a histogram with 10 bins for this attribute and save the plot in the plots/ directory
    plt.hist(column_data, bins = 10)
    plt.title("{} histogram".format(column_name))
    plt.xlabel("Attribute value")
    plt.ylabel("Counts")
    save_directory = os.path.join("plots", "{}_histogram.jpg".format(column_name))
    plt.savefig(save_directory)
    plt.figure()
