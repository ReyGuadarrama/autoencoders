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


tamanos_lote = [1, 32, 400]

configuracion_red = {
    'input': 2,
    'capas_ocultas': [4, 1],
    'activaciones': ['sigmoide', 'sigmoide'],
    'costo': 'bce',
    'optimizador': 'gd',
    'epocas': 20,
    'tamano_lote': 400,
    'lr': 0.1
}

resultados = []

for batch in tamanos_lote:
    configuracion_red['tamano_lote'] = batch
    parametros_entrenados, historial = rna.entrenar_red(X, T, configuracion_red)
    resultados.append(
        {
            'lote': batch,
            'parametros': parametros_entrenados,
            'historial': historial,
        }
    )

plt.figure()
for prueba in resultados:
    plt.plot(prueba['historial'], label=prueba['lote'])
plt.legend()


x1 = np.linspace(-3, 7, 100)
x2 = np.linspace(-6, 5, 100)
X1, X2 = np.meshgrid(x1, x2)
X_test = np.hstack((X1.reshape(-1, 1), X2.reshape(-1, 1)))

for prueba in resultados:
    plt.figure()
    prob_map = rna.predecir(X_test, prueba['parametros'], configuracion_red['activaciones'])
    plt.contourf(X1, X2, prob_map.reshape(X1.shape), cmap='cool', levels=50)
    plt.colorbar(label='probabilidad')
    plt.title(prueba['lote'])
    plt.scatter(X[:, 0], X[:, 1], c=T)

plt.show()




