import numpy as np
import matplotlib.pyplot as plt
import src.autoencoder as ae
from src.utils import generar_gausianas


modelo = np.load('modelos/prueba1.npz', allow_pickle=True)

parametros_entrenados = modelo['params'].item()
historial = modelo['historial']
configuracion_red = modelo['config'].item()

gausianas_prueba = generar_gausianas(5000)
t = np.linspace(-1,1,200)

rep_latente = ae.codificador(gausianas_prueba, parametros_entrenados, configuracion_red['activaciones'])
reconstruccion = ae.decodificador(rep_latente, parametros_entrenados, configuracion_red['activaciones'])


# Exploracion la organizacion del espacio latente

plt.figure()
plt.scatter(rep_latente[:,0],rep_latente[:,1], marker='.')
plt.show()

# plt.figure()
# for i in range(1):
#     n = np.random.randint(0, 1000)
#     plt.plot(t, gausianas_prueba[n], label='original')
#     plt.plot(t, reconstruccion[n], label='reconstruccion')

