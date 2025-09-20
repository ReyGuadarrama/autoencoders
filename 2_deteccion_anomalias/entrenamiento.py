import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.costos as costos

np.random.seed(37)

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


ondas_entrenamiento = generar_ondas(10000)

# datos de prueba
ondas_normales = generar_ondas(500)
ondas_anomalas = generar_ondas(500, rango_frec=(7.0, 9.0))

configuracion_red = {
    'input': 200,
    'capas_ocultas': [32, 16, 32, 200],
    'activaciones': ['relu', 'relu', 'relu', 'lineal'],
    'costo': 'mse',
    'optimizador': 'gd',
    'epocas': 2000,
    'tamano_lote': 32,
    'lr': 0.1
}

parametros_entrenados, historial = rna.entrenar_red(ondas_normales, ondas_normales, configuracion_red)

plt.figure()
plt.plot(historial[0], label='Entrenamiento')
plt.plot(historial[1], label='Validacion')
plt.legend()
plt.show()

np.savez('modelos/prueba.npz',
         params = parametros_entrenados,
         historial = historial,
         config =  configuracion_red)