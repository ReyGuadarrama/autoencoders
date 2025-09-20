import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna

modelo_relu = np.load('modelos/ruido-relu.npz', allow_pickle=True)

params_relu = {}
for i in modelo_relu:
    if i.startswith('W') or i.startswith('b'):
        params_relu[i] = modelo_relu[i]

reconstruccion_relu = rna.predecir()

