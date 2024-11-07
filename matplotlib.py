# Question 1
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import pandas as pd

# Load the Iris dataset
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target

species_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
iris_df['species'] = iris_df['species'].map(species_map)

# Visualization 1: Pairplot to show relationships between features, colored by species
sns.pairplot(iris_df, hue='species', palette='Set2', markers=["o", "s", "D"])
plt.suptitle('Pairplot of Iris Features by Species', y=1.02)
plt.show()

# Visualization 2: Boxplot to compare distributions of sepal length by species
plt.figure(figsize=(8, 6))
sns.boxplot(x='species', y='sepal length (cm)', data=iris_df, palette='Set2')
plt.title('Distribution of Sepal Length by Iris Species')
plt.show()

# Visualization 3: Violin plot to compare petal width across species
plt.figure(figsize=(8, 6))
sns.violinplot(x='species', y='petal width (cm)', data=iris_df, palette='Set2')
plt.title('Petal Width Distribution by Iris Species')
plt.show()


# Question 2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/LoanDataset - LoansDatasest.csv'
loan_data = pd.read_csv(file_path)

# Preprocessing: Remove currency symbols and convert loan amount and income columns to numeric
loan_data['loan_amnt'] = loan_data['loan_amnt'].replace('[£,]', '', regex=True).astype(float)
loan_data['customer_income'] = loan_data['customer_income'].replace('[£,]', '', regex=True).astype(float)

# Handle missing values in 'loan_int_rate' by filling with the median rate for simplicity
loan_data['loan_int_rate'].fillna(loan_data['loan_int_rate'].median(), inplace=True)

# Visualization 1: Distribution of loan amounts by loan intent
plt.figure(figsize=(10, 6))
sns.boxplot(data=loan_data, x='loan_intent', y='loan_amnt', palette="Set2")
plt.title('Loan Amount Distribution by Loan Intent')
plt.xlabel('Loan Intent')
plt.ylabel('Loan Amount (£)')
plt.xticks(rotation=45)
plt.show()

# Visualization 2: Loan interest rate distribution by loan grade
plt.figure(figsize=(10, 6))
sns.boxplot(data=loan_data, x='loan_grade', y='loan_int_rate', palette="coolwarm")
plt.title('Interest Rate Distribution by Loan Grade')
plt.xlabel('Loan Grade')
plt.ylabel('Interest Rate (%)')
plt.xticks(rotation=45)
plt.show()

# Visualization 3: Loan default rates by home ownership status
plt.figure(figsize=(10, 6))
sns.countplot(data=loan_data, x='home_ownership', hue='Current_loan_status', palette="pastel")
plt.title('Loan Status by Home Ownership')
plt.xlabel('Home Ownership')
plt.ylabel('Count')
plt.legend(title='Loan Status')
plt.show()
