import numpy as np
import src.activaciones as act
import src.costos as c
import src.optimizadores as op


def inicializar_RNA(capas):

    parametros = {}

    for i in range(len(capas)-1):
        neuronas_capa_actual = capas[i]
        neuronas_capa_siguiente = capas[i+1]

        parametros[f'W{i+1}'] = np.random.randn(neuronas_capa_actual, neuronas_capa_siguiente)
        parametros[f'b{i+1}'] = np.zeros((1, neuronas_capa_siguiente))

    return parametros


def propagacion_adelante(X, parametros, activaciones):
    valor_capas = {'a0': X}
    A = X

    for i, activacion in enumerate(activaciones):
        W = parametros[f'W{i+1}']
        b = parametros[f'b{i+1}']
        g, dg_dz = act.mapa_activaciones[activacion]

        Z = np.dot(A, W) + b
        A = g(Z)

        valor_capas[f'a{i+1}'] = A
        valor_capas[f'da/dz{i+1}'] = dg_dz(Z)

    return A, valor_capas

def retropropagacion(P, T, parametros, valor_capas, derivada_costo):
    L = len(parametros) // 2
    derivadas = {}

    delta = derivada_costo(T, P) * valor_capas[f'da/dz{L}']
    derivadas[f'dW{L}'] = np.matmul(valor_capas[f'a{L-1}'].transpose(), delta)
    derivadas[f'db{L}'] = np.sum(delta, axis=0, keepdims=True)

    for l in range(L-1, 0, -1):
        W = parametros[f'W{l+1}']
        delta = np.matmul(delta, W.transpose()) * valor_capas[f'da/dz{l}']
        derivadas[f'dW{l}'] = np.matmul(valor_capas[f'a{l-1}'].transpose(), delta)
        derivadas[f'db{l}'] = np.sum(delta, axis=0, keepdims=True)

    return derivadas

def actualizacion_parametros(X, T, config):
    capas = config['capas']
    activaciones = config['activaciones']
    costo_tipo = config['costo']
    optimizador_tipo = config['optimizador']
    epocas = config['epocas']
    lr = config['lr']

    # Obtener funciones de costo
    costo_fn, costo_derivada = c.mapa_costos[costo_tipo]
    optimizador_fn = op.mapa_optimizadores[optimizador_tipo]

    historial = np.zeros(epocas)

    parametros = inicializar_RNA(capas)

    for i in range(epocas):
        pred, val_capas = propagacion_adelante(X, parametros, activaciones)
        derivadas = retropropagacion(pred, T, parametros, val_capas, costo_derivada)
        parametros = optimizador_fn(parametros, derivadas, lr)

        historial[i] = costo_fn(T, pred)

    return parametros, historial


