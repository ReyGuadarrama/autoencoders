import numpy as np
import matplotlib.pyplot as plt
import src.red_neuronal as rna

# Extraccion de modelo entrenado

# Entrenamiento Relu
modelo_relu = np.load('modelos/relu.npz', allow_pickle=True)

parametros_relu = modelo_relu['params'].item()
historial_relu = modelo_relu['historial']
config_relu = modelo_relu['config'].item()

# Entrenamiento Sigmoide
modelo_sigmoide = np.load('modelos/sigmoide.npz', allow_pickle=True)

parametros_sigmoide = modelo_sigmoide['params'].item()
historial_sigmoide = modelo_sigmoide['historial']
config_sigmoide = modelo_sigmoide['config'].item()


# Definiciion de funciones
np.random.seed(37)

def generar_ondas(n_muestras):
    ondas = np.zeros((n_muestras, 200))
    t = np.linspace(0, 8*np.pi, 200)

    for i in range(n_muestras):
        frecuencia = np.random.uniform(3.0, 3.5)
        amplitud = np.random.uniform(0.5, 1.5)
        fase = np.random.uniform(0, 2*np.pi)
        frec_envolte = np.random.uniform(0.1, 0.3)

        envolvente = amplitud* np.sin(frec_envolte*t)
        ondas[i] = envolvente * np.sin(frecuencia * t + fase)

    return ondas

def agregar_ruido(ondas_limpias):
    ruido = np.random.normal(0.0, 0.3, ondas_limpias.shape)
    return ondas_limpias + ruido

ondas_limpias_prueba = generar_ondas(500)
ondas_ruidosas_prueba = agregar_ruido(ondas_limpias_prueba)

reconstruccion_relu = rna.predecir(ondas_ruidosas_prueba, parametros_relu, config_relu['activaciones'])
reconstruccion_sigmoide = rna.predecir(ondas_ruidosas_prueba, parametros_sigmoide, config_sigmoide['activaciones'])

plt.figure()
plt.plot(historial_relu[1], label='Costo ReLu')
plt.plot(historial_sigmoide[1], label='Costo Sigmoide')
plt.legend()


i = 10

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(ondas_limpias_prueba[i], label='onda original')
ax1.plot(reconstruccion_relu[i], label='reconstruccion')
ax1.plot(ondas_ruidosas_prueba[i], label='onda ruido', alpha=0.3)
ax1.set_title('Entrenamiento Relu')
ax1.legend()

ax2 = fig.add_subplot(212)
ax2.plot(ondas_limpias_prueba[i], label='onda original')
ax2.plot(reconstruccion_sigmoide[i], label='reconstruccion')
ax2.plot(ondas_ruidosas_prueba[i], label='onda ruido', alpha=0.3)
ax2.set_title('Entrenamiento Sigmoide')
ax2.legend()


plt.show()

