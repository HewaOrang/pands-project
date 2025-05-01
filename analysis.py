# analysis.py
# This program lists the codes used in the project.ipynb note book as requested by Andrew Beatty.
# Author: Hewa Orang

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

summary = df.describe(include='all') # 

with open("iris_summary.txt", "w") as f:
    f.write("Summary of Iris Dataset Variables\n\n")
    f.write(summary.to_string())

# Plotting.
import matplotlib.pyplot as plt # matplotlib is a plotting library

# Numerical arrays.
import numpy as np # NumPy is a library for numerical computing in Python

# Sepal length histogram
ax = df["sepal_length"].hist(grid=False, xlabelsize=10, ylabelsize=12, rwidth=0.9)
ax.set_title('Iris Data Features', weight='bold')
ax.set_xlabel('Sepal Length')
ax.set_ylabel('Frequency')
ax = plt.savefig('sepal_length')

# Sepal width histogram
ax = df["sepal_width"].hist(grid=False, xlabelsize=10, ylabelsize=12, rwidth=0.9)
ax.set_title('Iris Data Features', weight='bold')
ax.set_xlabel('sepal_width')
ax.set_ylabel('Frequency')
ax = plt.savefig('sepal_width')

# Petal length histogram
ax = df["petal_length"].hist(grid=False, xlabelsize=10, ylabelsize=12, rwidth=0.9)
ax.set_title('Iris Data Features', weight='bold')
ax.set_xlabel('petal_length')
ax.set_ylabel('Frequency')
ax = plt.savefig('petal_length')

# Petal width histogram
ax = df["petal_width"].hist(grid=False, xlabelsize=10, ylabelsize=12, rwidth=0.9)
ax.set_title('Iris Data Features', weight='bold')
ax.set_xlabel('petal_width')
ax.set_ylabel('Frequency')
ax = plt.savefig('petal_width')

# Combined histogram
plt.hist(df['sepal_length'], edgecolor='black', label='sepal_length', alpha=0.5)
plt.hist(df['sepal_width'], edgecolor='black', label='sepal_width', alpha=0.5)
plt.hist(df['petal_length'], edgecolor='black', label='petal_length', alpha=0.5)
plt.hist(df['petal_width'], edgecolor='black', label='petal_width', alpha=0.5)
plt.legend()
plt.savefig('all_features_histogram.png')

# Scatter plot: Sepal length vs Sepal width
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

fig, ax = plt.subplots()
for target, color, label in zip([0, 1, 2], ['black', 'green', 'yellow'], data.target_names):
    subset = df[df['target'] == target]
    ax.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'], c=color, label=label, marker='o')

ax.set_xlabel('Sepal_length')
ax.set_ylabel('Sepal_width')
ax.legend(title='classes')
plt.show()

# Scatter plot: Petal length vs Petal width
fig, ax = plt.subplots()
for target, color, label in zip([0, 1, 2], ['black', 'green', 'yellow'], data.target_names):
    subset = df[df['target'] == target]
    ax.scatter(subset['petal length (cm)'], subset['petal width (cm)'], c=color, label=label, marker='o')

ax.set_xlabel('petal_length')
ax.set_ylabel('petal_width')
ax.legend(title='classes')
plt.show()

# Box plots for petal lengths grouped by class
fig, ax = plt.subplots()
box_data = [df[df['target'] == i]['petal length (cm)'] for i in range(3)]
ax.boxplot(box_data)
ax.set_xticklabels(['Setosa', 'Versicolor', 'Virginica'])
ax.set_title('Petal Lengths by Class')
ax.set_ylabel('Petal Length (cm)')

# Correlation heatmap
import seaborn as sns

correlation_coeff = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_coeff, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Coefficients of Iris Dataset Features')
plt.show()