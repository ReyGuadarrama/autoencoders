import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.costos as costos
from src.utils import generar_ondas

np.random.seed(37)

modelo = np.load('modelos/prueba1.npz', allow_pickle=True)

parametros_entrenados = modelo['params'].item()
historial = modelo['historial']
configuracion_red = modelo['config'].item()

# datos de prueba
ondas_normales = generar_ondas(500)
ondas_anomalas = generar_ondas(500, rango_frec=(7.0, 9.0))

reconstruccion_normal = rna.predecir(ondas_normales, parametros_entrenados, configuracion_red['activaciones'])
reconstruccion_anomala = rna.predecir(ondas_anomalas, parametros_entrenados, configuracion_red['activaciones'])

mse_entrenamiento = costos.mse(ondas_normales, reconstruccion_normal)
mse_prueba = costos.mse(ondas_anomalas, reconstruccion_anomala)

fig, axes = plt.subplots(1, 4)
for i in range(4):
    n = np.random.randint(500)
    axes[i].plot(ondas_normales[n], 'r-', label='original normal')
    axes[i].plot(reconstruccion_normal[n], 'b--', label='re''construccion')

plt.title(f'datos normales {mse_entrenamiento:.4f}')
plt.legend()

fig, axes = plt.subplots(1, 4)
for i in range(4):
    n = np.random.randint(500)
    axes[i].plot(ondas_anomalas[n], label='original anomala')
    axes[i].plot(reconstruccion_anomala[n], label='reconstruccion')

plt.title(f'datos anomalos {mse_prueba:.4f}')
plt.legend()
plt.show()

ondas_prueba = np.vstack((ondas_normales, ondas_anomalas))
reconstruccion = rna.predecir(ondas_prueba, parametros_entrenados, configuracion_red['activaciones'])

umbral = 0.08
errores = np.mean((ondas_prueba - reconstruccion)**2, axis=1)
y = np.hstack([np.zeros(500), np.ones(500)])
deteccion = (errores > umbral).astype(float)

falsos_positivos = np.sum((deteccion==1) & (y==0))
falsos_negativos = np.sum((deteccion==0) & (y==1))
print(f'falsos positivos: {falsos_positivos}, falsos negativos: {falsos_negativos}')
acc = np.sum(y==deteccion)/1000
print(acc)

detecciones_erroneas = np.where((y!=deteccion))[0]
print(detecciones_erroneas)

fig, axes = plt.subplots(1, 4)
for i, index in enumerate(detecciones_erroneas[:4]):
    if errores[index] > umbral:
        det = 'falso positivo'
    else:
        det = 'falso negativo'

    axes[i].plot(ondas_prueba[index])
    axes[i].plot(reconstruccion[index])
    axes[i].set_title(f'mse = {errores[index]:.4f}, {det}')

plt.show()

