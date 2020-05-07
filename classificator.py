import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np

scaler = StandardScaler()

svm = pickle.load(open('svm_model.sav', 'rb'))
random_forest = pickle.load(open('random_forest_model.sav', 'rb'))
mlp = pickle.load(open('rna_model.sav', 'rb'))

novo_registro = [[50000, 40, 5000]]
novo_registro = np.asarray(novo_registro)
novo_registro = novo_registro.reshape(-1, 1)
novo_registro = scaler.fit_transform(novo_registro)
novo_registro = novo_registro.reshape(-1, 3)

resposta_svm = svm.predict(novo_registro)
resposta_random_forest = random_forest.predict(novo_registro)
resposta_mlp = mlp.predict(novo_registro)
