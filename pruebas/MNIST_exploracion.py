import matplotlib.pyplot as plt
import numpy as np

datos = np.load('../datos/mnist.npz')
X_entr, y_entr = datos['X_entr'], datos['y_entr']
X_prueba, y_prueba = datos['X_prueba'], datos['y_prueba']


print(X_entr.shape)
print(X_prueba.shape)

print(np.max(X_entr), np.min(X_entr))
print(np.max(X_prueba), np.min(X_prueba))

fig, axes = plt.subplots(4, 6)
for i, ax in enumerate(axes.flat):
    n = np.random.randint(10000)
    ax.imshow(X_entr[n].reshape(28,28), cmap='gray')
    ax.axis('off')

plt.show()
