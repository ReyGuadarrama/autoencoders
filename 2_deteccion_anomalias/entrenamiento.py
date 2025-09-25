import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.utils as utils

ondas_normales = utils.generar_ondas(10000)

configuracion_red = {
    'input': 200,
    'capas_ocultas': [50, 25, 50, 200],
    'activaciones': ['relu', 'relu', 'relu', 'lineal'],
    'costo': 'mse',
    'optimizador': 'gd',
    'epocas': 200,
    'tamano_lote': 50,
    'lr': 0.1
}

parametros_entrenados, historial = rna.entrenar_red(ondas_normales, ondas_normales, configuracion_red)

plt.figure()
plt.plot(historial[0], label='Entrenamiento')
plt.plot(historial[1], label='Validacion')
plt.legend()
plt.show()

np.savez('modelos/prueba1.npz',
         params=parametros_entrenados,
         historial=historial,
         config=configuracion_red)