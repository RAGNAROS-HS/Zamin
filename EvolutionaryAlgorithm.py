import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import seaborn.objects as so
from sklearn.model_selection import train_test_split  # Only for split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf

os.makedirs('output/ea', exist_ok=True)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

np.random.seed(42)
tf.random.set_seed(42)

# Load data
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv("housing.csv", header=None, delimiter=r"\s+", names=column_names)
target_col = 'MEDV'

X = data.drop(columns=[target_col])
y = data[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape}, Test: {X_test.shape}") 

mu = X_train.mean(axis=0)
sigma = X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma 
y_train = y_train.values
y_test = y_test.values


# NN topology (same as NeuralNetwork.py)
model = Sequential([
    Dense(22, activation='relu', input_shape=(13,)),  
    Dense(22, activation='relu'),                     
    Dense(22, activation='relu'),                     
    Dense(1, activation='linear')                     
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()


def decode_genotype(genotype):
    """Convert a flat genotype vector into the Keras weight format."""
    weights = []
    offset = 0
    for layer in model.layers:
        shapes = [w.shape for w in layer.get_weights()]
        for shape in shapes:
            size = np.prod(shape)
            weights.append(genotype[offset:offset + size].reshape(shape))
            offset += size
    return weights


def initialize_pop(population_size):
    x = []
    dims = sum(np.prod(w.shape) for w in model.get_weights())  
    print(f"Dimensions: {dims}")
    for i in range(population_size):
        x.append(np.random.randn(dims).astype(np.float64))
    return x


def fitness(genotype):
    X_np = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
    weights = decode_genotype(genotype)
    model.set_weights(weights)
    y_pred = model.predict(X_np, verbose=0).flatten()
    mse = np.mean((y_train - y_pred) ** 2)
    return 1.0 / (mse + 1e-10)  # higher = better

def selection(population, fitnesses, pressure):
    indices = random.sample(range(len(population)), k=3)  #need to remember about parameter tuning
    best_idx = indices[0]
    for idx in indices[1:]:
        if fitnesses[idx] > fitnesses[best_idx]:  # higher = better
            best_idx = idx
    return population[best_idx]
        


def mutation(individual, mutation_rate, sigma):
    mask = np.random.random(len(individual)) < mutation_rate
    individual[mask] += np.random.normal(0, sigma, size=mask.sum())
    return individual


def crossover(parent1, parent2, crossover_rate):
    mask = np.random.random(len(parent1)) < crossover_rate
    child1 = parent1.copy()
    child2 = parent2.copy()
    child1[mask] = parent2[mask]
    child2[mask] = parent1[mask]
    return child1, child2

def compute_metrics(y_true, y_pred):
    """Same metrics as NeuralNetwork.py."""
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R²': r2}


def ea_training_loop(epochs, population_size=100, tournament_size=3,
                     crossover_rate=0.5, mutation_rate=0.1, mutation_sigma=0.3,
                     elitism=2):

    population = initialize_pop(population_size)
    best_fitness_history = []
    avg_fitness_history = []

    for gen in range(epochs):
        fitnesses = [fitness(ind) for ind in population]

        best_fit = 1.0 / max(fitnesses)  
        avg_fit = np.mean([1.0 / f for f in fitnesses])
        best_fitness_history.append(best_fit)
        avg_fitness_history.append(avg_fit)

        #if gen % 20 == 0 or gen == epochs - 1:
        print(f"Gen {gen:>4d} | Best MSE: {best_fit:.4f} | Avg MSE: {avg_fit:.4f}")

        ranked = np.argsort(fitnesses)[::-1]  
        new_population = [population[ranked[i]].copy() for i in range(elitism)]

        while len(new_population) < population_size:
            p1 = selection(population, fitnesses, tournament_size)
            p2 = selection(population, fitnesses, tournament_size)
            c1, c2 = crossover(p1, p2, crossover_rate)
            c1 = mutation(c1, mutation_rate, mutation_sigma)
            c2 = mutation(c2, mutation_rate, mutation_sigma)
            new_population.append(c1)
            if len(new_population) < population_size:
                new_population.append(c2)

        population = new_population

    fitnesses = [fitness(ind) for ind in population]
    best_idx = int(np.argmax(fitnesses))
    best_genotype = population[best_idx]

    best_weights = decode_genotype(best_genotype)
    model.set_weights(best_weights)
    X_train_np = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
    X_test_np = X_test.values if hasattr(X_test, 'values') else np.array(X_test)
    y_pred_train = model.predict(X_train_np, verbose=0).flatten()
    y_pred_test = model.predict(X_test_np, verbose=0).flatten()

    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)

    print("\nEA Train Metrics:", train_metrics)
    print("EA Test Metrics:", test_metrics)

    plt.style.use('seaborn-v0_8-whitegrid')

    plt.figure(figsize=(10, 6))
    plt.plot(best_fitness_history, linewidth=2.5, color='steelblue', label='Best MSE')
    plt.plot(avg_fitness_history, linewidth=2.5, color='orange', alpha=0.6, label='Avg MSE')
    plt.title('EA Fitness Convergence')
    plt.xlabel('Generation')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('output/ea/ea_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.6)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('True MEDV')
    plt.ylabel('Predicted MEDV')
    plt.title(f'EA Test Predictions (R²={test_metrics["R²"]:.3f})')
    plt.savefig('output/ea/ea_test_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

    return best_genotype, best_fitness_history, train_metrics, test_metrics


best_genotype, history, train_metrics, test_metrics = ea_training_loop(
    epochs=50, population_size=100
)
