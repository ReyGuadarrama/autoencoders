import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.utils as utils
import src.costos as c


#Cargar modelo
modelo = np.load('modelos/prueba1.npz', allow_pickle=True)

parametros_entrenados = modelo['params'].item()
historial = modelo['historial']
configiracion_red = modelo['config'].item()

#Datos de prueba
ondas_normales = utils.generar_ondas(1000)
ondas_anomalas = utils.generar_ondas(1000, rango_frec=(8.0, 10.0))

reconstruccion_normal = rna.predecir(ondas_normales, parametros_entrenados, configiracion_red['activaciones'])
reconstruccion_anomala = rna.predecir(ondas_anomalas, parametros_entrenados, configiracion_red['activaciones'])

mse_normales = c.mse(ondas_normales, reconstruccion_normal)
mse_anomalias = c.mse(ondas_anomalas, reconstruccion_anomala)

ondas_prueba = np.vstack((ondas_normales, ondas_anomalas))
reconstruccion = rna.predecir(ondas_prueba, parametros_entrenados, configiracion_red['activaciones'])

umbral = 0.05
errores = np.mean((ondas_prueba - reconstruccion)**2, axis = 1)
y = np.hstack((np.zeros(1000), np.ones(1000)))
deteccion = (errores > umbral).astype(float)

falsos_positivos = np.sum((deteccion == 1) & (y == 0))
falsos_negativos = np.sum((deteccion == 0) & (y == 1))

print(f'Falsos positivos: {falsos_positivos}, Falsos negativos: {falsos_negativos}')
precision = np.sum(y==deteccion)/ 2000
print(f'Precision: {precision}')

detecciones_erroneas = np.where((y!=deteccion))[0]

fig, axes = plt.subplots(1, 4)
for i , index in enumerate(detecciones_erroneas[:4]):
    if errores[index] > umbral:
        det='Falso positivo'
    else:
        det='Falso negativo'

    axes[i].plot(ondas_prueba[index], color='blue', label ='original')
    axes[i].plot(reconstruccion[index], color='red', label ='reconstruccion')
    #axes[i].title(f'mse: {errores[index]}, {det}')

plt.show()
