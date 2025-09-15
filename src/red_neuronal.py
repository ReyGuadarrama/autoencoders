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

def generar_batches(X, T, batch_size):
    n_muestras = X.shape[0]
    indices = np.random.permutation(n_muestras)

    for i in range(0, n_muestras, batch_size):
        batch_indices = indices[i:i+batch_size]
        yield X[batch_indices], T[batch_indices]

def actualizar_parametros(X, T, parametros, config):
    activaciones = config['activaciones']
    costo_tipo = config['costo']
    optimizador_tipo = config['optimizador']
    batch_size = config.get('batch_size', X.shape[0])
    lr = config['lr']

    # Obtener funciones de costo
    costo_fn, costo_derivada = c.mapa_costos[costo_tipo]
    optimizador_fn = op.mapa_optimizadores[optimizador_tipo]

    costo_total = 0
    n_batches = 0

    for X_batch, T_batch in generar_batches(X, T, batch_size):
        pred, val_capas = propagacion_adelante(X_batch, parametros, activaciones)
        derivadas = retropropagacion(pred, T_batch, parametros, val_capas, costo_derivada)
        parametros = optimizador_fn(parametros, derivadas, lr)

        costo_total += costo_fn(T_batch, pred)
        n_batches += 1

    costo_promedio = costo_total / n_batches
    return parametros, costo_promedio

def entrenar_red(X, T, config):
    input = config['input']
    capas_ocultas = config['capas_ocultas']
    epocas = config['epocas']

    arquitectura = [input] + capas_ocultas
    parametros = inicializar_RNA(arquitectura)

    historial = np.zeros(epocas)

    for epoca in range(epocas):
        parametros, costo_epoca = actualizar_parametros(X, T, parametros, config)
        historial[epoca] = costo_epoca

    return parametros, historial

def predecir(X, parametros, activaciones):
    A = X

    for i, activacion in enumerate(activaciones):
        W = parametros[f'W{i + 1}']
        b = parametros[f'b{i + 1}']
        g, _ = act.mapa_activaciones[activacion]

        Z = np.dot(A, W) + b
        A = g(Z)

    return A