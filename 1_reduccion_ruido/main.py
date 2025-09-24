import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.costos as c

# Extraccion de modelo entrenado

# Entrenamiento MNIST
modelo = np.load('modelos/MNIST.npz', allow_pickle=True)

parametros = modelo['params'].item()
historial = modelo['historial']
config = modelo['config'].item()

# Carga de imagenes MNIST
datos = np.load('../datos/MNIST.npz')
X, y = datos['X'], datos['y']

def agregar_ruido(ondas_limpias):
    ruido = np.random.normal(0.0, 0.001, ondas_limpias.shape)
    return ondas_limpias + ruido

img_ruidosas = agregar_ruido(X)

reconstruccion = rna.predecir(X, parametros, config['activaciones'])
mse = c.mse(X, reconstruccion)
print(mse)

plt.figure()
plt.plot(historial[0], label='Costo ReLu')
plt.legend()

plt.figure()
plt.imshow(reconstruccion[0].reshape(28,28), cmap='gray')
plt.show()


print("X min/max:", X.min(), X.max())
print("Primeros 20 valores:", X[0][:20])

print("Reconstrucción min/max:", reconstruccion.min(), reconstruccion.max())
print("Primeros 20 valores:", reconstruccion[0][:20])


#fig, axes = plt.subplots(8, 6, figsize=(12, 16))

# for i, ax in enumerate(axes.reshape(-1, 2)):
#     n = np.random.randint(len(X))
#
#     # Original
#     ax[0].imshow(X[n].reshape(28,28), cmap="gray", vmin=0, vmax=1)
#     ax[0].set_title("Original", fontsize=8)
#     ax[0].axis("off")
#
#     # Reconstruida
#     ax[1].imshow(reconstruccion[n].reshape(28,28), cmap="gray", vmin=0, vmax=1)
#     ax[1].set_title("Reconstruida", fontsize=8)
#     ax[1].axis("off")
#
# plt.tight_layout()
# plt.show()


