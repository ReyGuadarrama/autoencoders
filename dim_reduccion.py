import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.autoencoder as ae

np.random.seed(37)

def generar_esfera():
    puntos = np.zeros((500, 3))
    for i in range(500):
        x1, x2 = np.random.uniform(low=-1, high=1, size=2)
        if x1**2 + x2**2 < 1:
            x = 2 * x1 * np.sqrt(1 - x1 ** 2 - x2 ** 2)
            y = 2 * x2 * np.sqrt(1 - x1 ** 2 - x2 ** 2)
            z = 1 - 2 * (x1**2 +x2**2)
            puntos[i] = np.array([x, y, z])

    return puntos

esfera_puntos = generar_esfera()

esfera_aum = np.hstack((esfera_puntos, esfera_puntos*2))
print(esfera_aum.shape)

configuracion_red = {
    'input': 6,
    'capas_ocultas': [3, 6],
    'activaciones': ['lineal', 'lineal'],
    'costo': 'mse',
    'optimizador': 'gd',
    'epocas': 500,
    'tamano_lote': 5000,
    'lr': 0.1
}

parametros_entrenados, historial = rna.entrenar_red(esfera_aum, esfera_aum, configuracion_red)

rep_latente = ae.codificador(esfera_aum, parametros_entrenados, configuracion_red['activaciones'])
reconstruccion = ae.decodificador(rep_latente, parametros_entrenados, configuracion_red['activaciones'])

plt.figure()
plt.plot(historial[0])

fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(esfera_puntos[:,0], esfera_puntos[:,1], esfera_puntos[:,2], c=esfera_puntos[:,2])
ax1.scatter(esfera_aum[:,3], esfera_aum[:,4], esfera_aum[:,5], c=esfera_aum[:,5], alpha=0.5, cmap='cool')
ax1.set_aspect('equal')

ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(rep_latente[:,0], rep_latente[:,1], rep_latente[:,2], c=esfera_puntos[:,2])
ax2.set_aspect('equal')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(reconstruccion[:,0], reconstruccion[:,1], reconstruccion[:,2], c=reconstruccion[:,2])
ax3.scatter(reconstruccion[:,3], reconstruccion[:,4], reconstruccion[:,5], c=reconstruccion[:,5], alpha=0.5, cmap='cool')
ax3.set_aspect('equal')
plt.show()