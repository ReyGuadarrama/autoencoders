import numpy as np

def generar_ondas(n_muestras, rango_frec=(2.8, 3.2)):
    ondas = np.zeros((n_muestras, 200))
    t = np.linspace(0, 8*np.pi, 200)

    for i in range(n_muestras):
        frecuencia = np.random.uniform(*rango_frec)
        amplitud = np.random.uniform(0.5, 1.5)
        fase = np.random.uniform(0, 2*np.pi)
        frec_envolvente = np.random.uniform(0.1, 0.3)

        envolvente = amplitud * np.sin(frec_envolvente * t)
        ondas[i] = envolvente *  np.sin(frecuencia * t + fase)

    return ondas

def generar_gausianas(n_muestras):
    gausianas = np.zeros((n_muestras, 200))
    t = np.linspace(-10, 10, 200)

    for i in range(n_muestras):
        mean = np.random.uniform(-1.0, 1.0)
        std = np.random.uniform(0.2, 2.0)

        onda = np.exp(-0.5 * ((t - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        gausianas[i] = onda

    return gausianas