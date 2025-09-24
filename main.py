import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2*0.1 #8*np.sin(x) + x**2

def df(x):
    return 0.2*x**2 #8*np.cos(x) + 2*x

x = np.linspace(-10, 10, 200)

w_gd = 8
w_gdm = 8
alfa = 0.01
v = 0
beta = 0.85
iteraciones = 50
costos = np.zeros((2, iteraciones))

fig, ax = plt.subplots()
for i in range(iteraciones):
    ax.clear()
    v = beta*v + alfa*df(w_gdm)
    w_gdm = w_gdm - v
    w_gd = w_gd - alfa*df(w_gd)
    c_gdm = f(w_gdm)
    c_gd = f(w_gd)
    costos[0, i] = c_gdm
    costos[1, i] = c_gd
    ax.plot(x, f(x))
    ax.scatter(w_gd, c_gd, color='red', zorder=2, label='GD')
    ax.scatter(w_gdm, c_gdm, color='purple', zorder=2, label='GDM')
    ax.set_title(f'Iteracion: {i}')
    plt.legend()
    plt.pause(0.2)


plt.figure()
plt.plot(costos[0], label='gd')
plt.plot(costos[1], label='gdm')
plt.legend()
plt.show()

