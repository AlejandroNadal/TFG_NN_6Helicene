import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Lambda


# 1) Load Rmax_mean & Rmax_std

stats_df = pd.read_csv('Rmax_stats.csv')  # Ensure this path is correct
Rmax_mean = float(stats_df['Rmax_abs_mean'][0])
Rmax_std  = float(stats_df['Rmax_abs_std'][0])
print(f">>> Using Rmax_mean = {Rmax_mean:.6f}, Rmax_std = {Rmax_std:.6f}")


# 2) Load Quantile Models → Build model_mean & model_sigma

model_lower = load_model('model_lower.h5', compile=False)   # predicts Q₀.₁₂
model_upper = load_model('model_upper.h5', compile=False)   # predicts Q₀.₈₈

inp = Input(shape=(16 * 5,), name='one_hot_input')
y_l = model_lower(inp)   # shape = (None, 1)
y_u = model_upper(inp)   # shape = (None, 1)

mean_out  = Lambda(lambda x: 0.5 * (x[0] + x[1]),  name='mean_output')([y_l, y_u])
sigma_out = Lambda(lambda x: 0.5 * (x[1] - x[0]), name='sigma_output')([y_l, y_u])

model_mean  = Model(inputs=inp, outputs=mean_out,  name='helicene_mean_model')
model_sigma = Model(inputs=inp, outputs=sigma_out, name='helicene_sigma_model')

# Inference only
def batch_mean_sigma(genotypes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    genotypes: shape = (batch_size, 16), dtype=int in {0..4}.
    Returns:
      mu_batch    = ½ (lower + upper),   shape = (batch_size,)
      sigma_batch = ½ (upper - lower),   shape = (batch_size,)
    """
    batch_size = genotypes.shape[0]
    X = np.zeros((batch_size, 16 * 5), dtype=np.float32)
    for i in range(batch_size):
        g = genotypes[i]
        for p, atom_code in enumerate(g):
            X[i, 5*p + atom_code] = 1.0

    y_l = model_lower.predict(X, batch_size=batch_size, verbose=0).squeeze()
    y_u = model_upper.predict(X, batch_size=batch_size, verbose=0).squeeze()
    mu_batch    = 0.5 * (y_l + y_u)
    sigma_batch = 0.5 * (y_u - y_l)
    return mu_batch, sigma_batch


# 3) GA Hyperparameters & Helper Functions

n_pos        = 16
n_atom_types = 5  # [H, F, Cl, Br, I]

pop_size        = 2048
num_generations = 300
tournament_k    = 3
crossover_rate  = 0.9
mutation_rate   = 0.15
elitism_size    = 2
stall_limit     = 30
λ_uncertainty   = 1.0 # λ controls the trade-off between exploration and exploitation in the acquisition function. Used for n > 6. due to the uncertainty in the model predictions, we want to balance between exploiting high mean values and exploring uncertain regions.

def random_individual_multi(n: int) -> np.ndarray:
    g = np.zeros((n_pos,), dtype=int) # Helicene full of Hydrogens (0)
    hal_pos = random.sample(range(n_pos), n) # hal_pos, e.g. [0, 3, 5, 7, 12] for n=5
    for i in hal_pos:
        g[i] = random.choice([1, 2, 3, 4]) # g[i] = 1, 2, 3, or 4 (F, Cl, Br, I), e.g. [0, 3, 5, 7, 12] -> [3, 2, 1, 4, 3]
    return g

def initialize_population(n: int) -> np.ndarray:
    pop = np.zeros((pop_size, n_pos), dtype=int)
    for i in range(pop_size):
        pop[i] = random_individual_multi(n) # Creates pop_size=2048 random individuals with exactly n halogens via random_individual_multi(n)
    return pop

def tournament_selection(pop: np.ndarray, fitnesses: np.ndarray, k: int = 3) -> np.ndarray:
    pop_s = pop.shape[0]
    contenders = random.sample(range(pop_s), k) # Randomly select k indices from the population, been an indice one of the 2048 individuals
    best_idx = max(contenders, key=lambda idx: fitnesses[idx]) # where key=lambda idx: fitnesses[idx] means we select the index of the individual with the highest fitness among the contenders
    return pop[best_idx].copy() # This returns the genotype of the best individual among the contenders

def uniform_crossover_multi(p1: np.ndarray, p2: np.ndarray, n: int) -> np.ndarray: # where p1 and p2 are two parent genotypes, n is the number of halogens
    child = np.zeros_like(p1)
    for i in range(n_pos):
        child[i] = p1[i] if (random.random() < 0.5) else p2[i]
    nonzero_idx = [i for i, v in enumerate(child) if v != 0]
    k = len(nonzero_idx)
    if k > n:
        to_zero = random.sample(nonzero_idx, k - n) # Basically if the child has more than n halogens, we randomly select k-n indices to set to 0
        for idx in to_zero:
            child[idx] = 0 # Set the selected indices to 0
    elif k < n:
        zero_idx = [i for i, v in enumerate(child) if v == 0]
        to_fill = random.sample(zero_idx, n - k)
        for idx in to_fill:
            child[idx] = random.choice([1, 2, 3, 4])
    return child

def mutate_multi(g: np.ndarray, n: int) -> np.ndarray:
    child = g.copy() # g.copy() creates a copy of the genotype to mutate
    if random.random() < 0.5:
        ones = [i for i, v in enumerate(child) if v != 0] # Find indices of halogen atoms (non-zero values) e.g. [0, 3, 5, 7, 12] for n=5
        zeros = [i for i, v in enumerate(child) if v == 0] # Find indices of Hydrogen atoms (zero values) e.g. [1, 2, 4, 6, 8, 9, 10, 11, 13, 14, 15] for n=5
        if ones and zeros:
            i = random.choice(ones) # Randonmly select and index from ones (halogen atoms) e.g. 3
            j = random.choice(zeros) # Randomly select an index from zeros (Hydrogen atoms) e.g. 1
            child[j] = child[i] # Set the selected Hydrogen atom to the value of the selected halogen atom e.g. if child[3] = 2 (Cl), then child[1] = 2
            child[i] = 0 # Set the selected halogen atom to 0 (Hydrogen) e.g. child[3] = 0
    else:
        hal_sites = [i for i, v in enumerate(child) if v != 0]
        if hal_sites:
            i = random.choice(hal_sites)
            current = child[i]
            choices = [t for t in [1, 2, 3, 4] if t != current] # Choices are F, Cl, Br, I excluding the current atom type meaning if current = 2 (Cl), then choices = [1, 3, 4] (F, Br, I)
            child[i] = random.choice(choices)
    return child

def evaluate_fitness(population: np.ndarray, n: int) -> np.ndarray:
    mu_vals, sigma_vals = batch_mean_sigma(population)
    if n <= 6:
        return mu_vals # For n <= 6, the fitness is just the mean μ, since we don't need to account for uncertainty because the model is accurate enough.
    else:
        return mu_vals - λ_uncertainty * sigma_vals # This is the acquisition function: μ - λ·σ for n > 6, where λ is a hyperparameter controlling the trade-off between exploration and exploitatition.


# 4) Run_ga_for_n

def run_ga_for_n(n: int, return_full_sorted: bool = False):
    """
    Run GA to optimize exactly n halogens.  
    If return_full_sorted=True, also return the entire final generation
    (sorted by fitness descending), along with their normalized μ and σ.
    """
    population = initialize_population(n)
    fitnesses  = evaluate_fitness(population, n)

    best_overall = None
    best_fit     = -np.inf
    stall_count  = 0 # Counts generations without improvement

    for gen in range(1, num_generations + 1):
        # Sort current generation by fitness (descending)
        idx_sorted = np.argsort(fitnesses)[::-1]
        population = population[idx_sorted]
        fitnesses  = fitnesses[idx_sorted]

        # Track overall best
        if fitnesses[0] > best_fit:
            best_fit     = fitnesses[0]
            best_overall = population[0].copy()
            stall_count  = 0
        else:
            stall_count += 1

        # Early stop if no improvement
        if stall_count >= stall_limit:
            break

        # Create next generation with elitism (keep best individuals) + selection/crossover/mutation
        new_pop = []
        for e in range(elitism_size):
            new_pop.append(population[e].copy()) # Keep the best `elitism_size` individuals unchanged

        while len(new_pop) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_k)
            if random.random() < crossover_rate:
                parent2 = tournament_selection(population, fitnesses, tournament_k)
                child   = uniform_crossover_multi(parent1, parent2, n)
            else:
                child = parent1.copy()

            if random.random() < mutation_rate:
                child = mutate_multi(child, n)

            new_pop.append(child)

        population = np.vstack(new_pop)
        fitnesses  = evaluate_fitness(population, n)

    # At this point, `population` and `fitnesses` are the final generation, already sorted.
    mu_vals, sigma_vals = batch_mean_sigma(population)
    # If n <= 6, fitnesses == mu_vals; if n > 6, fitnesses == (mu_vals - λ·sigma_vals). So everything is sorted.

    if return_full_sorted:
        # Return the single "best" plus the full arrays for post‐processing
        return (
            best_overall,      # best_genotype (1 × 16)
            mu_vals[0],        # normalized μ of best
            sigma_vals[0],     # normalized σ of best
            population,        # final pop sorted by fitness  (pop_size × 16)
            mu_vals,           # array of length pop_size (normalized μ)
            sigma_vals,        # array of length pop_size (normalized σ)
            fitnesses          # array of length pop_size (acquisition = μ or μ−λ·σ)
        )
    else:
        return best_overall, mu_vals[0], sigma_vals[0]


# 5) Main: Loop over n=7…16, collect Top 100 into a DataFrame, write CSV

if __name__ == "__main__":
    random.seed(2025)
    np.random.seed(2025)
    tf.random.set_seed(2025)

    # We will store a row for each of the Top 100 candidates at each n (7..16).
    # Columns:   n, rank (1..100), genotype (as a string), mu_real, sigma_real
    rows = []

    for n in range(7, 17):
        print(f"\n>>> Running GA for n = {n} (collecting Top 100) …")
        (
          best_genotype,
          mu_best_norm,
          sigma_best_norm,
          sorted_pop,
          sorted_mu_norm,
          sorted_sigma_norm,
          sorted_fitness
        ) = run_ga_for_n(n, return_full_sorted=True)

        # sorted_pop is shape (pop_size, 16), already sorted by fitness descending.
        # Take only the Top 100
        top100_genos    = sorted_pop[:100]      # shape = (100, 16)
        top100_mu_norm  = sorted_mu_norm[:100]  # shape = (100,)
        top100_sigma_norm = sorted_sigma_norm[:100]  # shape = (100,)

        # Convert each normalized μ and σ into real units:
        mu_real_list    = top100_mu_norm   * Rmax_std + Rmax_mean
        sigma_real_list = top100_sigma_norm * Rmax_std

        # For each of these 100, create a row:
        for rank_idx in range(100):
            geno = top100_genos[rank_idx]               # array of length 16
            mu_r = mu_real_list[rank_idx]
            s_r  = sigma_real_list[rank_idx]

            # Convert genotype array [0,4,3,…] into a simple string, e.g. "0-4-3-0-…-0"
            geno_str = "-".join(str(int(x)) for x in geno.tolist())

            rows.append({
                'n_subs': n,
                'rank': rank_idx + 1,       # 1 = best, 2 = 2nd best, etc.
                'genotype': geno_str,
                'mu_real': mu_r,
                'sigma_real': s_r
            })

    # Build a DataFrame containing all Top‐100 entries for n=7..16
    df_top100 = pd.DataFrame(rows, columns=['n_subs', 'rank', 'genotype', 'mu_real', 'sigma_real'])

    # Save it to CSV
    output_csv = 'ga_top100_n7_16.csv'
    df_top100.to_csv(output_csv, index=False)
    print(f"\n>>> Saved Top 100 GA results for n=7..16 to:  {output_csv}")

    # If you also want to run GA for a single n (e.g. n=7) and display its best:
    # best7, mu7_norm, sigma7_norm = run_ga_for_n(7, return_full_sorted=False)
    # mu7 = mu7_norm * Rmax_std + Rmax_mean
    # sigma7 = sigma7_norm * Rmax_std
    # print(f"Best (n=7) = {best7}, Predicted Rmax = {mu7:.2f} ± {sigma7:.2f}") 
