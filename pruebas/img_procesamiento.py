import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# pixeles = np.array([[255, 0],
#                     [80, 200]])
#
# plt.figure()
# plt.imshow(pixeles)
# plt.show()

img = imread('../datos/lemur.jpg')

# R = img[:,:,0]
# G = img[:,:,1]
# B = img[:,:,2]
#
# fig, axes = plt.subplots(1, 3)
#
# axes[0].imshow(np.stack((R,np.zeros_like(R), np.zeros_like(R)),axis=2))
# axes[1].imshow(np.stack((np.zeros_like(G), G, np.zeros_like(G)),axis=2))
# axes[2].imshow(np.stack((np.zeros_like(B), np.zeros_like(B), B),axis=2))

# plt.figure()
# plt.imshow(img)
# plt.show()

# Convertir imagen a blanco y negro
img_byn = np.mean(img, axis = 2)
dpi = 100
plt.figure(figsize=(427/dpi, 640/dpi), dpi=dpi)
plt.imshow(img_byn, cmap='gray')
plt.axis('off')
#
# # Seccionar imagen
# img_recortada = img_byn[100:300, 300:500]
# plt.figure()
# plt.imshow(img_recortada, cmap = 'gray')
# plt.axis('off')
# plt.show()

# Reduccion de resolucion en la imagen
# factor = 10
# alto, ancho = img_byn.shape
# nuevo_alto, nuevo_ancho = alto//factor, ancho//factor
# img_reducida = np.zeros((nuevo_alto, nuevo_ancho))
#
# for i in range(nuevo_alto):
#     for j in range(nuevo_ancho):
#         fila_i, fila_f = i*factor, (i+1)*factor
#         col_i, col_f = j*factor, (j+1)*factor
#
#         bloque = img_byn[fila_i:fila_f, col_i:col_f]
#         img_reducida[i, j] = np.mean(bloque)
#
#
# plt.figure()
# dpi = 100
# plt.figure(figsize=(nuevo_ancho/dpi, nuevo_alto/dpi), dpi=dpi)
# plt.imshow(img_reducida, cmap='gray')
# plt.show()
# # plt.figure()
# # plt.imshow(img_reducida, cmap='gray')
# # plt.axis('off')
# # plt.show()
#
# print(img_byn.shape)
# print(img_reducida.shape)

fig, axes = plt.subplots(5, 5)


subimagenes = []
for i in range(5):
    ancho = img_byn.shape[1]
    sub_ancho = ancho // 5
    for j in range(5):
        alto = img_byn.shape[0]
        sub_alto = alto // 5
        sub_imagen = img_byn[j * sub_alto:(j + 1) * sub_alto, i*sub_ancho:(i + 1) * sub_ancho]
        subimagenes.append(sub_imagen)
        axes[j, i].imshow(sub_imagen, cmap='gray')
        axes[j, i].axis('off')


X = np.array(subimagenes)

plt.figure()
plt.imshow(X[3], cmap='gray')
plt.show()


X_entrenamiento = X.reshape(25, -1)
print(X.shape)
print(X_entrenamiento.shape)
