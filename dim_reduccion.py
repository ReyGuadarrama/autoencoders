import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.autoencoder as ae

np.random.seed(37)

def generar_esfera():
    puntos = np.zeros((500, 3))
    for i in range(500):
        x1, x2 = np.random.uniform(-1, 1, 2)
        if x1**2 + x2**2 < 1:
            x = 2 * x1 * np.sqrt(1 - x1**2 - x2**2)
            y = 2 * x2 * np.sqrt(1 - x1**2 - x2**2)
            z = 1 - 2 * (x1**2 + x2**2)
            puntos[i] = np.array([x, y, z])

    return np.array(puntos)

esfera_puntos = generar_esfera()

red_config = {
    'input': 3,
    'capas_ocultas': [2, 3],
    'activaciones': ['lineal', 'lineal'],
    'costo': 'mse',
    'optimizador': 'gd',
    'lr': 0.2,
    'epocas': 200
}
param_entrenados, historial = rna.entrenar_red(X=esfera_puntos,
                                             T=esfera_puntos,
                                             config=red_config)

proyeccion_2D = ae.encoder(esfera_puntos, param_entrenados, red_config['activaciones'])
reconstruccion = ae.decoder(proyeccion_2D, param_entrenados, red_config['activaciones'])

fig = plt.figure()
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(esfera_puntos[:,0], esfera_puntos[:,1], esfera_puntos[:,2],
           c=esfera_puntos[:,2], cmap='cool', s=20)
ax1.set_aspect('equal')

ax2 = fig.add_subplot(132)
ax2.scatter(proyeccion_2D[:,0], proyeccion_2D[:,1],
            c=esfera_puntos[:,2], cmap='cool', s=20)
ax2.set_aspect('equal')

ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(reconstruccion[:,0], reconstruccion[:,1], reconstruccion[:,2],
            c=reconstruccion[:,2], cmap='cool', s=20)
ax3.set_aspect('equal')
plt.show()

plt.figure()
plt.plot(historial)
plt.show()
