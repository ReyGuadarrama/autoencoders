import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
from src.utils import generar_ondas

ondas_entrenamiento = generar_ondas(10000)

# datos de prueba
ondas_normales = generar_ondas(500)
ondas_anomalas = generar_ondas(500, rango_frec=(7.0, 9.0))

configuracion_red = {
    'input': 200,
    'capas_ocultas': [50, 16, 50, 200],
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

np.savez('modelos/prueba1.npz',
         params = parametros_entrenados,
         historial = historial,
         config =  configuracion_red)