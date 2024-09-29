import numpy as np
from scipy.interpolate import interp1d
from icecream import ic
import matplotlib.pyplot as plt

def find_closest_points(x, xi):
    """
    Находит четыре ближайшие точки к xi в списке x.
    """
    closest_points = []
    for i in range(len(x)):
        if x[i] >= xi:
            if i <= 1:
                closest_indices = list(range(4))
            elif i >= len(x) - 2:
                closest_indices = list(range(len(x)-4, len(x)))
            else:
                closest_indices = list(range(i-2, i+2))
            closest_points = [x[idx] if idx >= 0 and idx < len(x) else closest_points[-1] for idx in closest_indices]
            break
    ic(closest_points)
    return closest_points

def divided_diff(x, y):
    """
    Вычисление разделённых разностей.
    """
    n = len(y)
    coef = [0] * n
    coef[0] = y[0]

    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            if x[i] == x[i - j]:
                coef[i] = y[i]  # Просто присвоить значение y[i], чтобы избежать деления на ноль
            else:
                y[i] = (y[i] - y[i - 1]) / (x[i] - x[i - j])
                coef[i] = y[i]

    return coef

def newton_interpolation(x, y, xi):
    """
    Интерполяция методом Ньютона.
    """
    closest_points = find_closest_points(x, xi)
    x_interpolate = closest_points
    y_interpolate = [y[x.index(x_interpolate[0])], y[x.index(x_interpolate[1])], y[x.index(x_interpolate[2])], y[x.index(x_interpolate[3])]]
    ic(y_interpolate)
    coef = divided_diff(x_interpolate, y_interpolate)
    n = len(coef) - 1
    result = coef[n]

    for i in range(n - 1, -1, -1):
        result = result * (xi - x_interpolate[i]) + coef[i]

    return result

def Cx(xi):
    x = [0, 0, 0, 0, 0.8, 1, 1.2, 1.5, 2, 2.25, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
         19, 20, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41]
    y = [0.75, 0.75, 0.75, 0.75, 1, 1.15, 1.35, 1.6, 1.75, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6,
         1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6,
          1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6, 1.6]
    return newton_interpolation(x, y, xi/340)

def Get_ro(R): # В основной функции всё в метрах, в полиноме в километрах
    x = [100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54,
         52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19,
         18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    y = [7.890 * 10 ** (-5), 1.347 * 10 ** (-4), 0.0002, 0.0004, 0.0007, 0.0012, 0.0019, 0.0031, 0.0049, 0.0077,
         0.0119, 0.0178, 0.0266, 0.0393, 0.0578, 0.0839, 0.1210, 0.1729, 0.2443, 0.3411, 0.4694, 0.6289,
         0.8183, 1.0320, 1.2840, 1.5940, 1.9670, 2.4260, 2.9850, 3.6460, 4.4040, 5.2760, 6.2740, 7.4200,
         8.7040, 9.4060, 10.1500, 10.9300, 11.7700, 12.6500, 13.5900, 14.5700, 15.6200, 16.7100, 17.8800,
         19.1100, 20.3900, 21.7400, 23.1800, 24.6800, 26.2700, 27.9500, 29.7400, 31.6000, 33.5400,
         35.5800, 37.7200, 39.9500, 42.2600, 44.7100, 47.2400, 49.8700, 52.6200, 55.4700, 58.4500, 61.5600,
         64.7900]
    ro = newton_interpolation(x, y, R /1000)
    ic(ro)# после 100к полином работает некорректно
    return ro


CX = []
r = []
RO = []
T = []
t = 0
xi = 11000
R = 95_000

while R >= 0:
    ro = Get_ro(R)
    ic(ro, R, R/1000)
    R -= 100
    RO.append(ro)
    r.append(R)

plt.plot(RO, r)
plt.show()

while xi >= 0:

    Cxa = Cx(xi)
    ic("Cx :", Cxa, xi)
    xi -= 10
    ic(xi)
    CX.append(Cxa)
    T.append(xi)
plt.plot(T, CX)
plt.show()

'''x = [-0.55, -0.14, 0.27, 0.68, 1.09, 1.5, 1.91, 2.32, 2.73]
y = [2.374, 4.213, 4.986, 4.132, 4.128, 2.615, 1.877, 1.684, 0.219]'''

