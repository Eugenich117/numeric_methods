import numpy
from icecream import ic

def find_closest_points(x, xi):
    """
    Находит две ближайшие точки к xi в списке x.
    """
    closest_points = []
    for i in range(len(x)):
        if x[i] >= xi:
            if i == 0:
                closest_points = [x[i], x[i+1]]
            else:
                closest_points = [x[i-1], x[i]]
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
    ic(n)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            y[i] = (y[i] - y[i - 1]) / (x[i] - x[i - j])
            coef[i] = y[i]
    ic(coef)
    return coef

def newton_interpolation(x, y, xi):
    """
    Интерполяция методом Ньютона.
    """
    closest_points = find_closest_points(x, xi)
    x_interpolate = closest_points
    y_interpolate = [y[x.index(x_interpolate[0])], y[x.index(x_interpolate[1])]]

    coef = divided_diff(x_interpolate, y_interpolate)
    n = len(coef) - 1
    result = coef[n]

    for i in range(n - 1, -1, -1):
        result = result * (xi - x_interpolate[i]) + coef[i]

    return result
#РАБОТАЕТ КОРРЕКТНО УРАААААА
# Пример использования
x = [-0.55, -0.14, 0.27, 0.68, 1.09, 1.5, 1.91, 2.32, 2.73]
y = [2.374, 4.213, 4.986, 4.132, 4.128, 2.615, 1.877, 1.684, 0.219]
xi = 1.721
print("Интерполированное значение:", newton_interpolation(x, y, xi))

 # В основной функции всё в метрах, в полиноме в километра1