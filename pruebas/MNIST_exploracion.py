import matplotlib.pyplot as plt
import numpy as np

datos = np.load('../datos/MNIST.npz')
X, y = datos['X'], datos['y']
X = X / 255.0  # normalizar

fig, axes = plt.subplots(4, 6)
for i, ax in enumerate(axes.flat):
    n = np.random.randint(10000)
    ax.imshow(X[n].reshape(28,28), cmap='gray')
    ax.axis('off')

plt.show()
