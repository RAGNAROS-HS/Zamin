import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.model_selection import train_test_split  # Only for split

np.random.seed(seed=42)

# Load data
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housing.csv", header=None, delimiter=r"\s+", names=column_names)
target_col = 'MEDV'

X = data.drop(columns=[target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}") 




def initialize_pop(population_size):
    x = []
    dims = len(column_names) - 1
    for i in range(population_size):
        x.append(np.random.randn(dims).astype(np.float64))
    return x


pop = initialize_pop(10)

def fitness(genotype):
    return 0

def selection(population, pressure):
    tournament = 
        
    return 0
        


def mutation(individual):

    return 0


def crossover():
    return 0