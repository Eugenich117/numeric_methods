import matplotlib.pyplot as plt
'''
def divided_differences(x_values, y_values, k):
    result = 0
    for j in range(k + 1):
        mul = 1
        for i in range(k + 1):
            if i != j:
                mul *= x_values[j] - x_values[i]
        result += y_values[j]/mul
    return result


def create_Newton_polynomial(x_values, y_values):
    div_diff = []
    for i in range(1, len(x_values)):
        div_diff.append(divided_differences(x_values, y_values, i))

    def newton_polynomial(x):
        result = y_values[0]
        for k in range(1, len(y_values)):
            mul = 1
            for j in range(k):
                mul *= (x-x_values[j])
            result += div_diff[k-1]*mul
        return result
    return newton_polynomial

x_values = [100, 98, 96, 94, 92, 90, 88, 86, 84, 82, 80, 78, 76, 74, 72, 70, 68, 66, 64, 62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

y_values = [7.890*10**(-5), 1.347*10**(-4), 0.0002, 0.0004, 0.0007, 0.0012, 0.0019, 0.0031, 0.0049, 0.0077, 0.0119, 0.0178, 0.0266, 0.0393, 0.0578, 0.0839, 0.1210, 0.1729, 0.2443, 0.3411, 0.4694, 0.6289, 0.8183, 1.0320, 1.2840, 1.5940, 1.9670, 2.4260, 2.9850, 3.6460, 4.4040, 5.2760, 6.2740, 7.4200, 8.7040, 9.4060, 10.1500, 10.9300, 11.7700, 12.6500, 13.5900, 14.5700, 15.6200, 16.7100, 17.8800, 19.1100, 20.3900, 21.7400, 23.1800, 24.6800, 26.2700, 27.9500, 29.7400, 31.6000, 33.5400, 35.5800, 37.7200, 39.9500, 42.2600, 44.7100, 47.2400, 49.8700, 52.6200, 55.4700, 58.4500, 61.5600, 64.7900]
for i  in range(len(x_values)):
    print(x_values[i], y_values[i])
# Интерполируем плотность для новых точек

new_pol = create_Newton_polynomial(x_values, y_values)
# Визуализируем результаты
plt.figure(figsize=(10, 6))
#plt.plot(x_values, y_values, label='Исходные данные')
plt.plot(x_values, y_values)
plt.show()
plt.xlabel('Высота')
plt.ylabel('Плотность')
plt.title('Интерполяция плотности по высоте')
plt.legend()
plt.grid(True)
plt.show()


for x in x_values:
    print("x = {:.4f}\t y = {:.4f}".format(x, new_pol(x)))
'''
def divided_differences(x_values, y_values, k):
    result = 0
    for j in range(k + 1):
        mul = 1
        for i in range(k + 1):
            if i != j:
                mul *= x_values[j] - x_values[i]
        result += y_values[j]/mul
    return result


def create_Newton_polynomial(x_values, y_values):
    div_diff = []
    for i in range(1, len(x_values)):
        div_diff.append(divided_differences(x_values, y_values, i))
    def newton_polynomial(x):
        result = y_values[0]
        for k in range(1, len(y_values)):
            mul = 1
            for j in range(k):
                mul *= (x-x_values[j])
            result += div_diff[k-1]*mul
        return result
    return newton_polynomial


x_values = [0, 2, 3, 5]
y_values = [0, 3, 5, 2]

new_pol = create_Newton_polynomial(x_values, y_values)

for x in x_values:
    print("x = {:.4f}\t y = {:.4f}".format(x, new_pol(x)))
for x in x_values:
    print("x = {:.4f}\t y = {:.4f}".format((new_pol(x)), new_pol(y)))