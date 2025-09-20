import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.costos as costos

np.random.seed(37)

modelo = np.load('modelos/prueba.npz', allow_pickle=True)

parametros_entrenados = modelo['params'].item()
historial = modelo['historial']
configuracion_red = modelo['config'].item()

def generar_ondas(n_muestras, rango_frec=(2.8, 3.2)):
    ondas = np.zeros((n_muestras, 200))
    t = np.linspace(0, 8*np.pi, 200)

    for i in range(n_muestras):
        frecuencia = np.random.uniform(*rango_frec)
        amplitud = np.random.uniform(0.5, 1.5)
        fase = np.random.uniform(0, 2*np.pi)
        frec_envolvente = np.random.uniform(0.1, 0.3)

        envolvente = amplitud * np.sin(frec_envolvente * t)
        ondas[i] = envolvente *  np.sin(frecuencia * t + fase)

    return ondas

# datos de prueba
ondas_normales = generar_ondas(500)
ondas_anomalas = generar_ondas(500, rango_frec=(7.0, 9.0))

reconstruccion_normal = rna.predecir(ondas_normales, parametros_entrenados, configuracion_red['activaciones'])
reconstruccion_anomala = rna.predecir(ondas_anomalas, parametros_entrenados, configuracion_red['activaciones'])

mse_entrenamiento = costos.mse(ondas_normales, reconstruccion_normal)
mse_prueba = costos.mse(ondas_anomalas, reconstruccion_anomala)

umbral = 0.01
errores_anomalos = np.mean((ondas_anomalas - reconstruccion_anomala)**2, axis=1)
errores_normales = np.mean((ondas_normales - reconstruccion_normal)**2, axis=1)
errores = np.hstack([errores_anomalos, errores_normales])
y = np.hstack([np.ones(500), np.zeros(500)])
deteccion = (errores > umbral).astype(int)

acc = np.sum(y==deteccion)/1000
print(acc)

detecciones_erroneas = np.where((y==deteccion)==0)[0]
print(detecciones_erroneas)

fig, axes = plt.subplots(1, 4)
for i, index in enumerate(detecciones_erroneas[:4]):
    axes[i].plot(ondas_normales[index-500])
    axes[i].plot(reconstruccion_normal[index-500])
    axes[i].set_title(f'mse = {errores[index]:.4f}')


# fig, axes = plt.subplots(1, 4)
# for i in range(4):
#     n = np.random.randint(500)
#     axes[i].plot(ondas_normales[n])
#     axes[i].plot(reconstruccion_normal[n])
#
# plt.title(f'datos normales {mse_entrenamiento:.4f}')
#
# fig, axes = plt.subplots(1, 4)
# for i in range(4):
#     n = np.random.randint(500)
#     axes[i].plot(ondas_anomalas[n])
#     axes[i].plot(reconstruccion_anomala[n])
#
# plt.title(f'datos anomalos {mse_prueba:.4f}')

plt.show()

