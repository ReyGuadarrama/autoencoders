import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.autoencoder as ae

def agregar_ruido(ondas_limpias):
    ruido = np.random.normal(0.0, 0.5, ondas_limpias.shape)
    return ondas_limpias + ruido

datos = np.load('../datos/MNIST.npz')
X, y = datos['X'], datos['y']

img_ruidosas = agregar_ruido(X)


configuracion_red = {
    'input': 784,
    'capas_ocultas': [64, 784],
    'activaciones': ['relu', 'sigmoide'],
    'costo': 'mse',
    'optimizador': 'gdm',
    'epocas': 200,
    'tamano_lote': 250,
    'lr': 0.4
}

parametros_entrenados, historial = rna.entrenar_red(img_ruidosas, X, configuracion_red)

plt.figure()
plt.plot(historial[0], label='Entrenamiento')
plt.plot(historial[1], label='Validacion')
plt.legend()

reconstruccion = rna.predecir(X, parametros_entrenados, configuracion_red['activaciones'])

fig, axes = plt.subplots(4, 6)

for i, ax_pair in enumerate(axes.reshape(-1, 3)):
    n = np.random.randint(len(X))

    ax_pair[0].imshow(img_ruidosas[n].reshape(28, 28), cmap="gray")
    ax_pair[0].set_title("Ruidosa", fontsize=8)
    ax_pair[0].axis("off")

    ax_pair[1].imshow(reconstruccion[n].reshape(28, 28), cmap="gray")
    ax_pair[1].set_title("Reconstruida", fontsize=8)
    ax_pair[1].axis("off")

    ax_pair[2].imshow(X[n].reshape(28, 28), cmap="gray")
    ax_pair[2].set_title("Original", fontsize=8)
    ax_pair[2].axis("off")

plt.show()

np.savez('modelos/MNIST.npz',
         params=parametros_entrenados,
         historial=historial,
         config=configuracion_red)



