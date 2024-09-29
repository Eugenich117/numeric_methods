from icecream import ic
import scipy
import numpy as np
import matplotlib.pyplot as plt


def dV_func(x, y):
    dy = (2*x**2 - 2*x*y) / (x**2 + 1)
    return dy


def runge_kutta_4(x, y, dt):
    k1 = dV_func(x, y)
    k2 = dV_func(x + dt/2, y + k1 * dt / 2)
    k3 = dV_func(x + dt/2, y + k2 * dt / 2)
    k4 = dV_func(x + dt, y + k3 * dt)
    y += (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    return y

X_SOLVE = []
Y_SOLVE = []
# Начальные условия
x0 = -2
x_solve = -2
y0 = -0.93333333333
dt = 0.2

while x_solve <= 4.2:
    y_solve = (2 / 3) * ((x_solve**3 + 1) / (x_solve**2 + 1))
    print(x_solve, y_solve)
    X_SOLVE.append(x_solve)
    Y_SOLVE.append(y_solve)
    x_solve += 0.2

# Выполнение метода Рунге-Кутты для заданного диапазона x
x_values = np.arange(x0, 4.2, dt)
y_values = np.zeros_like(x_values)
y_values[0] = y0


for i in range(1, len(x_values)):
    y_values[i] = runge_kutta_4(x_values[i-1], y_values[i-1], dt)


# Вывод результатов
for x, y in zip(x_values, y_values):
    print(f"x = {x:.2f}, y = {y:.4f}")


plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values, '-', color='blue')  # Первый график синего цвета
plt.plot(X_SOLVE, Y_SOLVE, '--', color='red')  # Второй график красного цвета
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()







