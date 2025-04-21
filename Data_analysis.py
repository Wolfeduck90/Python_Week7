# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Load and Explore the Dataset

# Load the dataset
# Replace 'iris.csv' with the path to your dataset
df = pd.read_csv("iris.csv")

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Explore the structure of the dataset
print("\nData types and missing values:")
print(df.info())

# Check for missing values
print("\nMissing values count:")
print(df.isnull().sum())

# Clean the dataset by dropping rows with missing values (if any)
df_cleaned = df.dropna()

print("\nDataset after cleaning (no missing values):")
print(df_cleaned.info())

# Task 2: Basic Data Analysis

# Compute basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
print(df_cleaned.describe())

# Group data by 'species' and compute mean petal length for each group
grouped_data = df_cleaned.groupby('species')['petal_length'].mean()
print("\nMean petal length for each species:")
print(grouped_data)

# Identify patterns or interesting findings
# Example: Comparing petal length and petal width correlation
correlation = df_cleaned['petal_length'].corr(df_cleaned['petal_width'])
print("\nCorrelation between petal length and petal width:", correlation)

# Task 3: Data Visualization

# Line Chart: Trends in petal length (example data sorted by petal length)
df_sorted = df_cleaned.sort_values(by='petal_length')
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['petal_length'], label="Petal Length Trend")
plt.title("Line Chart: Petal Length Trend")
plt.xlabel("Index")
plt.ylabel("Petal Length")
plt.legend()
plt.show()

# Bar Chart: Comparison of average petal length across species
plt.figure(figsize=(10, 6))
grouped_data.plot(kind='bar', color=['#ff9999','#66b3ff','#99ff99'])
plt.title("Bar Chart: Average Petal Length per Species")
plt.xlabel("Species")
plt.ylabel("Average Petal Length")
plt.xticks(rotation=0)
plt.show()

# Histogram: Distribution of petal length
plt.figure(figsize=(10, 6))
plt.hist(df_cleaned['petal_length'], bins=20, color='#5cb85c', edgecolor='black')
plt.title("Histogram: Distribution of Petal Length")
plt.xlabel("Petal Length")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Relationship between petal length and petal width
plt.figure(figsize=(10, 6))
plt.scatter(df_cleaned['petal_length'], df_cleaned['petal_width'], color='#d9534f')
plt.title("Scatter Plot: Petal Length vs. Petal Width")
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.show()

# Observations
print("\nObservations:")
print("- The petal length and petal width show a positive correlation.")
print("- Setosa species has smaller petal lengths compared to Versicolor and Virginica.")
print("- The distribution of petal lengths shows clusters indicating distinct groups.")
