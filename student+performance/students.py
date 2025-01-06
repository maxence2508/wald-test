import pandas as pd
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Charger les fichiers CSV avec le séparateur ';'
mat = pd.read_csv('student-mat.csv', sep=';')
por = pd.read_csv('student-por.csv', sep=';')

# Concaténer les deux fichiers en un seul DataFrame
students = pd.concat([mat, por], ignore_index=True)

# Séparer les groupes : avec et sans Internet
wi = students[students['internet'] == 'yes']
wo = students[students['internet'] == 'no']

# Calcul du nombre d'élèves dans chaque groupe
n_wi = len(wi)
n_wo = len(wo)

# Calcul des moyennes et écarts-types des échecs (failures)
m_wi = wi['failures'].mean()
s_wi = wi['failures'].std()

m_wo = wo['failures'].mean()
s_wo = wo['failures'].std()

# Afficher les résultats
print(f"Nombre d'élèves avec Internet : {n_wi}")
print(f"Nombre d'élèves sans Internet : {n_wo}")
print(f"Moyenne des échecs (avec Internet) : {m_wi:.2f}")
print(f"Écart-type des échecs (avec Internet) : {s_wi:.2f}")
print(f"Moyenne des échecs (sans Internet) : {m_wo:.2f}")
print(f"Écart-type des échecs (sans Internet) : {s_wo:.2f}")

# Test de Wald
diff_means = m_wi-m_wo
stat_observed = np.abs(diff_means)/np.sqrt(s_wi**2/n_wi+s_wo**2/n_wo)
p_value = 1-norm.cdf(stat_observed)
print("p-valeur = ", p_value)
