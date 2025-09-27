import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna
import src.utils as utils
import src.costos as c


#Cargar modelo
modelo = np.load('modelos/MNIST_prueba1.npz', allow_pickle=True)

parametros_entrenados = modelo['params'].item()
historial = modelo['historial']
configiracion_red = modelo['config'].item()

#Datos de prueba
img_MNIST = np.load('../datos/MNIST.npz')
X = img_MNIST['X']
y = img_MNIST['y']
img_unos = X[(y==1)]
img_ceros = X[(y==0)]

reconstruccion_normal = rna.predecir(img_ceros, parametros_entrenados, configiracion_red['activaciones'])
reconstruccion_anomala = rna.predecir(img_unos, parametros_entrenados, configiracion_red['activaciones'])

mse_normales = c.mse(img_ceros, reconstruccion_normal)
mse_anomalias = c.mse(img_unos, reconstruccion_anomala)

img_prueba = np.vstack((img_ceros, img_unos))
reconstruccion = rna.predecir(img_prueba, parametros_entrenados, configiracion_red['activaciones'])

print(mse_normales)
print(mse_anomalias)

umbral = 0.07
errores = np.mean((img_prueba - reconstruccion)**2, axis = 1)
y = np.hstack((np.zeros(img_ceros.shape[0]), np.ones(img_unos.shape[0])))
deteccion = (errores > umbral).astype(float)

falsos_positivos = np.sum((deteccion == 1) & (y == 0))
falsos_negativos = np.sum((deteccion == 0) & (y == 1))

print(f'Falsos positivos: {falsos_positivos}, Falsos negativos: {falsos_negativos}')
precision = np.sum(y==deteccion)/ (img_ceros.shape[0]+img_unos.shape[0])
print(f'Precision: {precision}')

detecciones_erroneas = np.where((y!=deteccion))[0]

fig, axes = plt.subplots(4, 6)
for i , ax in enumerate(axes.reshape(-1, 2)):
    index = np.random.choice(detecciones_erroneas)
    if errores[index] > umbral:
        det='Falso positivo'
    else:
        det='Falso negativo'

    ax[0].imshow(img_prueba[index].reshape(28, 28), cmap='gray')
    ax[0].axis("off")
    ax[1].imshow(reconstruccion[index].reshape(28, 28), cmap='gray')
    ax[1].axis("off")
    ax[0].set_title(f'mse: {errores[index]:.3f}, {det}')

plt.show()
