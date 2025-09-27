import numpy as np
import matplotlib.pyplot as plt
import src.autoencoder as ae
from src.utils import generar_gausianas


modelo = np.load('modelos/prueba1.npz', allow_pickle=True)

parametros_entrenados = modelo['params'].item()
historial = modelo['historial']
configuracion_red = modelo['config'].item()

gausianas_prueba = generar_gausianas(5000)
t = np.linspace(-10,10,200)

rep_latente = ae.codificador(gausianas_prueba, parametros_entrenados, configuracion_red['activaciones'])
reconstruccion = ae.decodificador(rep_latente, parametros_entrenados, configuracion_red['activaciones'])


# Exploracion la organizacion del espacio latent
medias = np.linspace(-1, 1, 10)
desviaciones = np.linspace(1.0, 2.5, 10)

plt.figure()
plt.scatter(rep_latente[:,0],rep_latente[:,1], marker='.')
for media, std in zip(medias, desviaciones):
    media_fija = generar_gausianas(500, media=(media, media))
    std_fija = generar_gausianas(500, desviacion=(std, std))
    media_fija_latente = ae.codificador(media_fija, parametros_entrenados, configuracion_red['activaciones'])
    std_fija_latente = ae.codificador(std_fija, parametros_entrenados, configuracion_red['activaciones'])
    plt.scatter(media_fija_latente[:,0],media_fija_latente[:,1])
    plt.scatter(std_fija_latente[:,0],std_fija_latente[:,1], label=f'{std:.3f}')
plt.legend()
plt.show()

# Exploracion de la continuidad del espacio latente
medias = np.linspace(-0.3, 0.7, 10)
desviaciones = np.linspace(1.0, 1.6, 10)
caracteristicas_ondas = np.stack([medias, desviaciones], axis=1)
trayectoria = []

fig, axes = plt.subplots(1, 2)
# Definir colormap
cmap = plt.get_cmap('spring')

# Scatter base
axes[0].scatter(rep_latente[:,0], rep_latente[:,1], marker='.', alpha=0.3, color='royalblue')

n = len(caracteristicas_ondas)

for i, (media, std) in enumerate(caracteristicas_ondas):
    trayectoria_ondas = generar_gausianas(1, media=(media, media), desviacion=(std, std))
    trayectoria_latente = ae.codificador(trayectoria_ondas, parametros_entrenados, configuracion_red['activaciones'])

    # Obtener color directamente del colormap, normalizando con división
    color = cmap(i / (n-1))

    # Usar el mismo color en scatter y plot
    axes[0].scatter(
        trayectoria_latente[:,0],
        trayectoria_latente[:,1],
        color=color, edgecolor='black', linewidth=1.0
    )
    axes[1].plot(t, trayectoria_ondas[0], color=color)

plt.show()


