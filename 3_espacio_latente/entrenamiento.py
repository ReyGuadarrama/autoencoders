import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
from src.utils import generar_gausianas

gausianas = generar_gausianas(10000)
t = np.linspace(-1,1,200)


configuracion_red = {
    'input': 200,
    'capas_ocultas': [100, 25, 2, 25, 100, 200],
    'activaciones': ['relu', 'relu', 'sigmoide', 'relu', 'relu', 'lineal'],
    'costo': 'mse',
    'optimizador': 'gdm',
    'epocas': 200,
    'tamano_lote': 32,
    'lr': 0.3
}

parametros_entrenados, historial = rna.entrenar_red(gausianas, gausianas, configuracion_red)

plt.figure()
plt.plot(historial[0], label='Entrenamiento')
plt.plot(historial[1], label='Validacion')
plt.legend()
plt.show()

np.savez('modelos/prueba1.npz',
         params = parametros_entrenados,
         historial = historial,
         config =  configuracion_red)