import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.autoencoder as ae

np.random.seed(37)

def generar_ondas(n_muestras):
    ondas = np.zeros((n_muestras, 200))
    t = np.linspace(0, 8 * np.pi, 200)

    for i in range(n_muestras):
        frecuencia = np.random.uniform(3.0, 3.5)
        amplitud = np.random.uniform(0.5, 1.5)
        fase = np.random.uniform(0, 2 * np.pi)
        frec_envolte = np.random.uniform(0.1, 0.3)

        envolvente = amplitud * np.sin(frec_envolte * t)
        ondas[i] = envolvente * np.sin(frecuencia * t + fase)

    return ondas


def agregar_ruido(ondas_limpias):
    ruido = np.random.normal(0.0, 0.3, ondas_limpias.shape)
    return ondas_limpias + ruido


ondas_limpias = generar_ondas(10000)
ondas_ruidosas = agregar_ruido(ondas_limpias)

configuracion_red = {
    'input': 200,
    'capas_ocultas': [50, 25, 50, 200],
    'activaciones': ['relu', 'relu', 'relu', 'lineal'],
    'costo': 'mse',
    'optimizador': 'gd',
    'epocas': 5000,
    'tamano_lote': 50,
    'lr': 0.1
}

parametros_entrenados, historial = rna.entrenar_red(ondas_ruidosas, ondas_limpias, configuracion_red)

plt.figure()
plt.plot(historial)
plt.show()

np.savez('modelos/relu.npz',
         params=parametros_entrenados,
         historial=historial,
         config=configuracion_red)

#
# ondas_limpias_prueba = generar_ondas(500)
# ondas_ruidosas_prueba = agregar_ruido(ondas_limpias_prueba)
#
# reconstruccion = rna.predecir(ondas_ruidosas_prueba, parametros_entrenados, configuracion_red['activaciones'])
#
# plt.figure()
# plt.plot(historial)
# plt.show()
#
# plt.figure()
# plt.plot(ondas_limpias_prueba[10], label='onda original')
# plt.plot(reconstruccion[10], label='reconstruccion')
# plt.legend()
# plt.show()



