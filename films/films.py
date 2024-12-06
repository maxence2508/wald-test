import pandas as pd
import scipy.stats as stats
import numpy as np


# Remplacez par le chemin de votre fichier CSV
file_path = 'movie_metadata.csv'

# Charger le fichier CSV
df = pd.read_csv(file_path)

# Extraire le premier genre de chaque ligne
df['first_genre'] = df['genres'].str.split('|').str[0]

# Filtrer les films d'Action et d'Adventure en fonction du premier genre
action_films = df[df['first_genre'] == 'Action']
adventure_films = df[df['first_genre'] == 'Adventure']

# Calculer le nombre de films pour chaque genre
n_action = action_films.shape[0]
n_adventure = adventure_films.shape[0]

# Calculer la moyenne des num_voted_users pour chaque genre
mean_action_voters = action_films['num_voted_users'].mean()
mean_adventure_voters = adventure_films['num_voted_users'].mean()

# Calculer l'écart-type des num_voted_users pour chaque genre
std_action_voters = action_films['num_voted_users'].std()
std_adventure_voters = adventure_films['num_voted_users'].std()
print("std:")
print(std_action_voters, std_adventure_voters)

print("nb films d'action : ", n_action)
print("nb films d'aventure : ", n_adventure)

print(f"Moyenne des utilisateurs ayant voté pour les films Action : {mean_action_voters}")
print(f"Moyenne des utilisateurs ayant voté pour les films Adventure : {mean_adventure_voters}")

z_obs = mean_action_voters-mean_adventure_voters

# Statistique du test
stat_obs = np.abs(z_obs)/np.sqrt(std_action_voters**2/n_action + (std_adventure_voters)**2/n_adventure)

p_valeur = 1-stats.norm.cdf(stat_obs)
print("p-valeur = ", p_valeur)