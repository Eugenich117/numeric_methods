"""файл с заготовкой метода интегрирования для систем дифференциальных уравнений  с автоматическим
масштабированием под количество переменных и уравнений """
"""пример данных, подаваемых функции для корректной работы
dt = 0.01
initial = {}
initial['S'] = S
initial['mass'] = mass
dx = ['V', 'L', 'tetta', 'R']
equations = [dV_func, dL_func, dtetta_func, dR_func]
также хочу напомнить, что нельзя забывать переводить угловые величины в радианы перед решением системы уравнений и
переводить обратно для вывода в файл регистрации и на графики
выгллядит примерно так 

cToDeg = 180 / m.pi
cToRad = m.pi / 180

после задания в градусах угловой величины
tetta *= cToRad

после вычислений
TETTA.append(tetta * cToDeg) """


def runge_kutta_4(equations, initial, dt, dx):
    '''equations - это список названий функций с уравнениями для системы
    initial это переменные с начальными условиями
    dx - это список переменных, которые будут использованы для интегрирования уравнения'''
    k1 = {key: 0 for key in initial.keys()}
    k2 = {key: 0 for key in initial.keys()}
    k3 = {key: 0 for key in initial.keys()}
    k4 = {key: 0 for key in initial.keys()}

    derivatives_1 = {key: initial[key] for key in initial}
    derivatives_2 = {key: initial[key] for key in initial}
    derivatives_3 = {key: initial[key] for key in initial}
    derivatives_4 = {key: initial[key] for key in initial}

    new_values = [0] * len(equations)

    for i, eq in enumerate(equations):
        derivative, key = eq(initial)
        k1[key] += derivative
        derivatives_1[key] = initial[key] + derivative * dt / 2
        derivatives_1[dx[i]] += dt / 2
        # derivatives_1 = {key: value / 2 for key, value in derivatives_1.items()}

    for i, eq in enumerate(equations):
        derivative, key = eq(derivatives_1)
        k2[key] += derivative
        derivatives_2[key] = initial[key] + derivative * dt / 2
        derivatives_2[dx[i]] += dt / 2
        # derivatives_2 = {key: value / 2 for key, value in derivatives_2.items()}

    for i, eq in enumerate(equations):
        derivative, key = eq(derivatives_2)
        k3[key] += derivative
        derivatives_3[key] = initial[key] + derivative * dt
        derivatives_3[dx[i]] += dt

    for i, eq in enumerate(equations):
        derivative, key = eq(derivatives_3)
        k4[key] += derivative
        derivatives_4[key] = initial[key] + derivative * dt
        new_values[i] = initial[key] + (1 / 6) * dt * (k1[key] + 2 * k2[key] + 2 * k3[key] + k4[key])
    return new_values

"""прмер вызова фнукции
while R >= Rb:
    initial.update({'tetta': tetta, 'Cxa': Cxa, 'ro': ro, 'L': L, 'V': V, 'R': R})
    values = runge_kutta_4(equations, initial, dt, dx)
    V = values[0]
    L = values[1]
    tetta = values[2]
    R = values[3]
    t += dt"""
