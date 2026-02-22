import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs('output/dataanalysis', exist_ok=True)


#loading data and annotating columns
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housing.csv", header=None, delimiter=r"\s+", names=column_names)

#seaborn dataset visualization

sns.set_theme()
#for name in column_names:
plt.figure(figsize=(12, 10))  # Width x height in inches
sns.clustermap(data.corr(), annot=True, cmap='coolwarm', center=0, square=True)

plt.savefig("output/dataanalysis/corr_clustermap.png", dpi=600, bbox_inches='tight')
