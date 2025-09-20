import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.costos as costos

np.random.seed(37)

def generar_ondas(n_muestras):
    ondas = np.zeros((n_muestras, 200))
    t = np.linspace(0, 8*np.pi, 200)

    for i in range(n_muestras):
        frecuencia = np.random.uniform(3.0, 3.5)
        amplitud = np.random.uniform(0.5, 1.5)
        fase = np.random.uniform(0, 2*np.pi)
        frec_envolvente = np.random.uniform(0.1, 0.3)

        envolvente = amplitud * np.sin(frec_envolvente * t)
        ondas[i] = envolvente *  np.sin(frecuencia * t + fase)

    return ondas

def agregar_ruido(ondas_limpias):
    ruido = np.random.normal(0, 0.3, ondas_limpias.shape)
    return ondas_limpias + ruido

ondas_limpias = generar_ondas(500)
ondas_ruidosas = agregar_ruido(ondas_limpias)

configuracion_red = {
    'input': 200,
    'capas_ocultas': [64, 32, 16, 32, 64, 200],
    'activaciones': ['relu', 'relu', 'relu', 'relu', 'relu', 'lineal'],
    'costo': 'mse',
    'optimizador': 'gd',
    'epocas': 2000,
    'tamano_lote': 64,
    'lr': 0.3
}

parametros_entrenados, historial = rna.entrenar_red(ondas_ruidosas, ondas_limpias, configuracion_red)

ondas_limpias_prueba = generar_ondas(1000)
ondas_ruidosas_prueba = agregar_ruido(ondas_limpias_prueba)

reconstruccion = rna.predecir(ondas_ruidosas, parametros_entrenados, configuracion_red['activaciones'])
reconstruccion_prueba = rna.predecir(ondas_ruidosas_prueba, parametros_entrenados, configuracion_red['activaciones'])

mse_entrenamiento = costos.mse(ondas_limpias, reconstruccion)
mse_prueba = costos.mse(ondas_limpias_prueba, reconstruccion_prueba)

plt.figure()
plt.plot(historial[0], label='Entrenamiento')
plt.plot(historial[1], label='Validacion')

fig, axes = plt.subplots(1, 4)
for i in range(4):
    n = np.random.randint(500)
    axes[i].plot(ondas_limpias[n])
    axes[i].plot(reconstruccion[n])
    axes[i].plot(ondas_ruidosas[n], alpha=0.3)

plt.title(f'datos entrenamiento {mse_entrenamiento:.4f}')

fig, axes = plt.subplots(1, 4)
for i in range(4):
    n = np.random.randint(500)
    axes[i].plot(ondas_limpias_prueba[n])
    axes[i].plot(reconstruccion_prueba[n])
    axes[i].plot(ondas_ruidosas_prueba[n], alpha=0.3)

plt.title(f'datos prueba {mse_prueba:.4f}')

plt.show()


# np.savez('modelos/ruido-relu.npz',
#          **parametros_entrenados,
#          historial = historial,
#          config =  configuracion_red)