import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

pixeles = np.array([[255, 0],
                    [80, 160]])

plt.figure()
plt.imshow(pixeles, cmap='gray')
plt.show()

img = imread('../datos/lemur.jpg')

# plt.figure()
# plt.imshow(img)
#
# R = img[:,:,0]
# G = img[:,:,1]
# B = img[:,:,2]
#
# # Grafica la imagen separandola por canales
# fig, axes = plt.subplots(1,3)
#
# axes[0].imshow(np.stack([R, np.zeros_like(R), np.zeros_like(R)], axis=2))
# axes[0].set_title("Rojo")
#
# axes[1].imshow(np.stack([np.zeros_like(G), G, np.zeros_like(G)], axis=2))
# axes[1].set_title("Verde")
#
# axes[2].imshow(np.stack([np.zeros_like(B), np.zeros_like(B), B], axis=2))
# axes[2].set_title("Azul")

# grafica la imagen en blanco y negro
plt.figure()
img_byn = np.mean(img, axis=2)
img_byn = img_byn[2:,:]
plt.imshow(img_byn, cmap='gray')

# # Seleccion de seccion
# img_recortada = img_byn[100:300,300:500]
# plt.figure()
# plt.imshow(img_recortada, cmap='gray')

img_aumentada = np.vstack([img_byn, np.zeros((30, 640))])
plt.figure()
plt.imshow(img_aumentada, cmap='gray')
plt.show()

# Reduccion de dimension
factor = 10
alto, ancho = img_byn.shape
nuevo_alto, nuevo_ancho = alto//factor, ancho//factor
img_reducida = np.zeros((nuevo_alto, nuevo_ancho))

for i in range(nuevo_alto):
    for j in range(nuevo_ancho):
        fila_i, fila_f = i*factor, (i+1)*factor
        col_i, col_f = j*factor, (j+1)*factor

        bloque = img_byn[fila_i:fila_f, col_i:col_f]
        img_reducida[i, j] = np.mean(bloque)

plt.figure()
plt.imshow(img_reducida, cmap='gray')
plt.axis('off')
plt.show()

# Vectoriazacion
# img_prueba = np.array([[1, 2, 3],
#                        [4, 5, 6],
#                        [7, 8, 9]])
#
# print(img_prueba)
# img_prueba_vectorizada = img_prueba.reshape(1, -1)
# print(img_prueba_vectorizada)


# Seleccion sistematica
#img_byn = img_byn[2:,:]

fig, axes = plt.subplots(5,5)

X = np.array([])

subimagenes = []
for i in range(5):
    ancho = img_byn.shape[1]
    seccion = ancho // 5
    for j in range(5):
        alto = img_byn.shape[0]
        seccion2 = alto // 5
        img_seccion = img_byn[j * seccion2:(1 + j) * seccion2, i * seccion:(i + 1) * seccion]
        subimagenes.append(img_seccion)
        axes[j, i].imshow(img_seccion, cmap='gray')
        axes[j, i].axis('off')

#plt.show()

X = np.array(subimagenes)

plt.figure()
plt.imshow(X[2], cmap='gray')
plt.axis('off')
plt.show()


X_entrenamiento = X.reshape(25, -1)

plt.figure()
plt.imshow(X_entrenamiento[0].reshape(85, 128), cmap='gray') #85x128
plt.show()

