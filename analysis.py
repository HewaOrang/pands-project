# analysis.py
# This program lists the codes used in the project.ipynb notebook as requested by Andrew Beatty.
# Author: Hewa Orang

# analysis.py
# This program lists the codes used in the project.ipynb notebook as requested by Andrew Beatty.
# Author: Hewa Orang

# Task 1: Loading the dataset
# Data frames

import pandas as pd

# Machine Learning Library that contains example datasets
import sklearn as sk1

# Load the iris dataset from csv file.
df = pd.read_csv('https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv')

# Show data
df

# Load the dataset.
data = sk1.datasets.load_iris()

# Task 2: Output summary of each variable to a single text file.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html
summary = df.describe(include='all')  # Describe computes the summary statistics for each column

with open("iris_summary.txt", "w") as f:  # Opens a file named "iris_summary.txt" in write mode.
    f.write("Summary of Iris Dataset Variables\n\n")  # Writes a header to the file.
    f.write(summary.to_string())  # Writes the string representation of the summary to the file.

# Task 3: Histogram of each variable
# Plotting.
import matplotlib.pyplot as plt  # matplotlib is a plotting library

# Numerical arrays.
import numpy as np  # NumPy is a library for numerical computing in Python

# https://python-graph-gallery.com/528-customizing-histogram-with-pandas/#:~:text=Adding%20titles%20and%20names%20to,()%20)%20functions%20to%20add%20them.
# ax is the histogram object, creates a histogram for the specific feature column "sepal_length"
ax = df["sepal_length"].hist(grid=False,  # Remove grid
                             xlabelsize=10,  # Change size of labels on the x-axis
                             ylabelsize=12,  # Change size of labels on the y-axis
                             rwidth=0.9  # Space between bins
                             )
ax.set_title('Iris Data Features', weight='bold')

# Add label names
ax.set_xlabel('Sepal Length')
ax.set_ylabel('count')

# Show the plot
ax = plt.savefig('sepal_length')

# Task 4: Scatter plot of each pair of variables
# https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/legend.html
df = pd.DataFrame(data.data, columns=data.feature_names)  # this converts the data into a pandas DataFrame
df['target'] = data.target  # this adds the target column to the DataFrame

# Create a figure and an axis.
fig, ax = plt.subplots()

# Scatter plot.
for target, color, label in zip([0, 1, 2], ['black', 'green', 'yellow'], data.target_names):  # Iterate over three classes (0, 1, 2) (setosa, versicolor, virginica) in different colors.
    subset = df[df['target'] == target]  # This creates a subset of the DataFrame for each target class.
    ax.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],  # This filters the DF for features called 'sepal length (cm)' and 'sepal width (cm)'
               c=color, label=label, marker='o')

# Labels.
ax.set_xlabel('Sepal_length')
ax.set_ylabel('Sepal_width')

ax.legend(title='classes')
plt.show()

# Task 5: Box Plots
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.boxplot.html
# https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.set_xticklabels.html
fig, ax = plt.subplots()

# Create boxplot data grouped by class
# https://chatgpt.com/share/680376d7-0e34-8006-88eb-1004c392dd33
box_data = [df[df['target'] == i]['petal length (cm)'] for i in range(3)]  # This creates a list of petal lengths for each class (0, 1, 2).

# Create boxplot
ax.boxplot(box_data)

# Add class labels (0, 1, 2 typically correspond to setosa, versicolor, virginica)
ax.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'])

ax.set_title('Petal Lengths by Class')
ax.set_ylabel('Petal Length (cm)')

# Task 6: Compute Correlations
# Calculate the correlation coefficients between the features.
# Display the results as a heatmap using matplotlib.
# https://zion-oladiran.medium.com/exploratory-data-analysis-iris-dataset-68897497b120

import seaborn as sns  # Seaborn is a data visualization library based on Matplotlib

correlation_coeff = df.corr()  # This calculates the correlation coefficients between the features in the DataFrame. The result is a correlation matrix shown as heatmap below.
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_coeff, annot=True, cmap='coolwarm', fmt='.2f')  # This creates a heatmap of the correlation coefficients using seaborn.
plt.title('Correlation Coefficients of Iris Dataset Features')
plt.show()

# Task 7: Too Many Features
# Using seaborn to create a pairplot of the data set.
# https://seaborn.pydata.org/generated/seaborn.pairplot.html
# https://www.tutorialspoint.com/plotting-graph-for-iris-dataset-using-seaborn-and-matplotlib#:~:text=The%20pairplot()%20function%20from,on%20the%20species%20of%20Iris.

iris = sns.load_dataset('iris')

# Create the pairplot
sns.pairplot(iris, hue='species')  # hue is used to color the points by species

# Display the plot
plt.show()