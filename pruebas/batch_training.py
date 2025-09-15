import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna

np.random.seed(37)

def data_generator(media1, media2):
    x1 = np.random.normal(media1, 1, 200)
    x2 = np.random.normal(media2, 1, 200)

    X = np.vstack((x1, x2)).transpose()

    return X


X1 = data_generator(1.0, 2.0)
X2 = data_generator(3.0, -2.0)
y1 = np.zeros(200).reshape(-1, 1)
y2 = np.ones(200).reshape(-1, 1)

X = np.vstack((X1, X2))
T = np.vstack((y1, y2))

tamanos_batch = [1, 32, 400]

config = {
    'input': 2,
    'capas_ocultas': [4, 1],
    'activaciones': ['sigmoide', 'sigmoide'],
    'costo': 'bce',
    'optimizador': 'gd',
    'epocas': 100,
    'lr': 0.1
}


resultados = []

for batch in tamanos_batch:
    config['batch_size'] = batch
    parametros, historial = rna.entrenar_red(X, T, config)
    resultados.append(
        {
            'batch_size': config['batch_size'],
            'activaciones': config['activaciones'],
            'parametros': parametros,
            'historial': historial
        }
    )


plt.figure()
for prueba in resultados:
    plt.plot(prueba['historial'], label=f'tamano del batch {prueba['batch_size']}')
plt.legend()

x1 = np.linspace(-3, 7, 100)
x2 = np.linspace(-6, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
X_test = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))

for prueba in resultados:
    plt.figure()
    prob_map = rna.predecir(X_test, prueba['parametros'], prueba['activaciones'])
    plt.contourf(X1, X2, prob_map.reshape(X1.shape), cmap='cool', levels=50)
    plt.colorbar(label='probabilidad')
    plt.scatter(X[:, 0], X[:, 1], c=T)
    plt.title(f'tamano del batch {prueba['batch_size']}')

plt.show()
