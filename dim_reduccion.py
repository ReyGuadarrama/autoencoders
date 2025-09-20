import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.autoencoder as ae

np.random.seed(37)

def generar_esfera():
    puntos = np.zeros((5000, 3))
    for i in range(5000):
        x1, x2 = np.random.uniform(low=-1, high=1, size=2)
        if x1**2 + x2**2 < 1:
            x = 2 * x1 * np.sqrt(1 - x1 ** 2 - x2 ** 2)
            y = 2 * x2 * np.sqrt(1 - x1 ** 2 - x2 ** 2)
            z = 1 - 2 * (x1**2 +x2**2)
            puntos[i] = np.array([x, y, z])

    return puntos

esfera_puntos = generar_esfera()

configuracion_red = {
    'input': 3,
    'capas_ocultas': [2, 3],
    'activaciones': ['lineal', 'lineal'],
    'costo': 'mse',
    'optimizador': 'gd',
    'epocas': 100,
    'tamano_lote': 5000,
    'lr': 0.1
}

parametros_entrenados, historial = rna.entrenar_red(esfera_puntos, esfera_puntos, configuracion_red)

rep_latente = ae.codificador(esfera_puntos, parametros_entrenados, configuracion_red['activaciones'])
reconstruccion = ae.decodificador(rep_latente, parametros_entrenados, configuracion_red['activaciones'])

fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(esfera_puntos[:,0], esfera_puntos[:,1], esfera_puntos[:,2])
ax1.set_aspect('equal')

ax2 = fig.add_subplot(132)
ax2.scatter(rep_latente[:,0], rep_latente[:,1])
ax1.set_aspect('equal')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(reconstruccion[:,0], reconstruccion[:,1], reconstruccion[:,2])
ax1.set_aspect('equal')
plt.show()