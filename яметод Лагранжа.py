import numpy
import math
import matplotlib.pyplot as plt
#интерполирует сразу все значения, без указания Xi
initial_speed = 4900
angle = 30
angle_in_rad = math.radians(angle)


def create_basic_polynomial(x_values, i):
    def basic_polynomial(x):
        divider = 1
        result = 1
        for j in range(len(x_values)):
            if j != i:
                result *= (x-x_values[j])
                divider *= (x_values[i]-x_values[j])
        return result/divider
    return basic_polynomial


def create_Lagrange_polynomial(x_values, y_values):
    basic_polynomials = []
    for i in range(len(x_values)):
        basic_polynomials.append(create_basic_polynomial(x_values, i))
    def lagrange_polynomial(x):
        result = 0
        for i in range(len(y_values)):
            result += y_values[i]*basic_polynomials[i](x)
        return result
    return lagrange_polynomial


x_values = [
        100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52,
        50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
        14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0] 
y_values = [
        7.890*10**(-5), 1.347*10**(-4), 0.0002, 0.0004, 0.0007, 0.0012, 0.0019, 0.0031, 0.0049, 0.0077,
        0.0119, 0.0178, 0.0266, 0.0393, 0.0578, 0.0839, 0.1210, 0.1729, 0.2443, 0.3411, 0.4694, 0.6289, 0.8183,
        1.0320, 1.2840, 1.5940, 1.9670, 2.4260, 2.9850, 3.6460, 4.4040, 5.2760, 6.2740, 7.4200, 8.7040, 9.4060,
        10.1500, 10.9300, 11.7700, 12.6500, 13.5900, 14.5700, 15.6200, 16.7100, 17.8800, 19.1100, 20.3900, 21.7400,
        23.1800, 24.6800, 26.2700, 27.9500, 29.7400, 31.6000, 33.5400, 35.5800, 37.7200, 39.9500, 42.2600, 44.7100,
        47.2400, 49.8700, 52.6200, 55.4700, 58.4500, 61.5600, 64.7900]

lag_pol = create_Lagrange_polynomial(x_values, y_values)

plt.plot(x_values, y_values)
plt.show()
# Вычисление и вывод результата


