import numpy as np

def gradiente_descendente(parametros, derivadas, config):
    lr = config['lr']
    parametros_actualizados = {}
    for llave in parametros.keys():
        parametros_actualizados[llave] = parametros[llave] - lr * derivadas[f'd{llave}']
    return parametros_actualizados

def gradiente_descendente_momentum(parametros, derivadas, config):
    lr = config['lr']
    beta = config.get('beta', 0.9)

    if 'velocidades' not in config:
        config['velocidades'] = {}
        for llave in parametros.keys():
            config['velocidades'][llave] = np.zeros_like(parametros[llave])

    parametros_actualizados = {}
    for llave in parametros.keys():
        config['velocidades'][llave] = beta * config['velocidades'][llave] + lr * derivadas[f'd{llave}']
        parametros_actualizados[llave] = parametros[llave] - config['velocidades'][llave]

    return parametros_actualizados


mapa_optimizadores = {
    'gd': gradiente_descendente,
    'gdm': gradiente_descendente_momentum
}