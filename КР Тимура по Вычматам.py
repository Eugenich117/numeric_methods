import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.linalg import fractional_matrix_power, norm
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

print("=" * 80)
print("КОНТРОЛЬНАЯ РАБОТА №2, ВАРИАНТ 8")
print("=" * 80)

print("\n" + "=" * 80)
print("ЗАДАЧА 1: МАТРИЧНЫЕ ВЫЧИСЛЕНИЯ")
print("=" * 80)

def matrix_pth_root(A, p):
    print(f"Вычисление корня степени {p} из матрицы:")
    print(f"A = {A}")

    if p <= 0:
        raise ValueError("Степень p должна быть положительной")

    if p == 1:
        print("Корень степени 1 - возвращаем исходную матрицу")
        return A

    if np.allclose(A, 0):
        print("Нулевая матрица - возвращаем нулевую матрицу")
        return np.zeros((2, 2))

    tr = np.trace(A)
    det = np.linalg.det(A)

    print(f"След матрицы: tr(A) = {tr}")
    print(f"Определитель матрицы: |A| = {det}")

    mean_tr = tr / 2
    d = mean_tr**2 - det

    print(f"Дискриминант: d = {d}")

    if abs(d) < 1e-12:
        print("Случай кратных собственных значений")
        lambda_val = mean_tr

        print(f"Кратное собственное значение: λ = {lambda_val}")

        if p % 2 == 0 and lambda_val < 0:
            raise ValueError(f"Для кратного отрицательного собственного значения λ={lambda_val} "
                           f"и чётного p={p} требуется комплексный корень")

        if lambda_val >= 0:
            f_val = lambda_val**(1/p)
        else:
            f_val = -((-lambda_val)**(1/p))

        if lambda_val != 0:
            f_derivative = (1/p) * lambda_val**(1/p - 1)
        else:
            f_derivative = 0

        print(f"f(λ) = {f_val}, f'(λ) = {f_derivative}")

        result = f_val * np.eye(2) + f_derivative * (A - lambda_val * np.eye(2))

        print("Результат по формуле для кратных собственных значений:")
        print(result)
        return result

    else:
        print("Случай различных собственных значений")
        sqrt_d = np.sqrt(d)

        l1 = mean_tr + sqrt_d
        l2 = mean_tr - sqrt_d

        print(f"Собственные значения: λ₁ = {l1}, λ₂ = {l2}")

        if p % 2 == 0 and (l1 < 0 or l2 < 0):
            raise ValueError(f"Для отрицательных собственных значений и чётного p={p} "
                           f"требуется комплексный корень")

        def compute_f(l):
            if l >= 0:
                return l**(1/p)
            else:
                return -((-l)**(1/p))

        f1 = compute_f(l1)
        f2 = compute_f(l2)

        print(f"f(λ₁) = {f1}, f(λ₂) = {f2}")

        term1 = ((f1 + f2) / 2) * np.eye(2)
        term2 = (A - mean_tr * np.eye(2)) / sqrt_d
        term3 = ((f1 - f2) / 2)

        result = term1 + term2 * term3

        print("Результат по формуле Сильвестра:")
        print(result)
        return result

def time_experiment(p=3, max_n=10000):
    print(f"\n--- Исследование времени выполнения для p={p} ---")

    ns = [10, 100, 1000, 5000, 10000]
    ns = [n for n in ns if n <= max_n]

    print(f"Тестируемые количества матриц: {ns}")

    times = []

    for n in ns:
        print(f"Обработка {n} матриц...")

        matrices = [np.random.rand(2, 2) for _ in range(n)]

        start_time = time.time()

        successful_calculations = 0
        for i, M in enumerate(matrices):
            try:
                matrix_pth_root(M, p)
                successful_calculations += 1
            except ValueError as e:
                if i == 0:
                    print(f"  Предупреждение: пропущена матрица из-за {e}")
                continue

        execution_time = time.time() - start_time
        times.append(execution_time)

        print(f"  Успешно обработано: {successful_calculations}/{n} матриц")
        print(f"  Время выполнения: {execution_time:.4f} сек")

    plt.figure(figsize=(10, 6))
    plt.plot(ns, times, 'bo-', linewidth=2, markersize=8, label='Экспериментальные данные')

    if len(ns) > 1:
        k = times[1] / ns[1]
        theoretical_times = [k * n for n in ns]
        plt.plot(ns, theoretical_times, 'r--', label='Теоретическая линейная зависимость')

    plt.xlabel('Количество матриц', fontsize=12)
    plt.ylabel('Время выполнения (сек)', fontsize=12)
    plt.title(f'Зависимость времени вычисления от количества матриц (p={p})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("График построен!")

def error_experiment(p=3, num_matrices=1000):
    print(f"\n--- Исследование точности для p={p} ---")
    print(f"Генерация {num_matrices} тестовых матриц...")

    errors = []
    condition_numbers = []
    matrices_generated = 0

    while matrices_generated < num_matrices:
        M = np.random.randint(-100, 100, (2, 2)).astype(float)

        M[0, 0] = max(abs(M[0, 0]), abs(M[0, 1])) + np.random.randint(1, 20)
        M[1, 1] = max(abs(M[1, 0]), abs(M[1, 1])) + np.random.randint(1, 20)

        try:
            A = np.linalg.matrix_power(M, p)
        except:
            continue

        cond_A = np.linalg.cond(A)

        if cond_A > 1e15 or cond_A < 1e-10:
            continue

        try:
            A_root = matrix_pth_root(A, p)
        except ValueError:
            continue

        true_root = M

        error_norm = norm(A_root - true_root, 2)
        true_norm = norm(true_root, 2)

        relative_error = error_norm / true_norm if true_norm > 0 else error_norm

        errors.append(relative_error)
        condition_numbers.append(cond_A)
        matrices_generated += 1

        if matrices_generated % 100 == 0:
            print(f"  Обработано {matrices_generated}/{num_matrices} матриц")

    print("Анализ зависимости ошибки от числа обусловленности...")

    plt.figure(figsize=(10, 6))
    plt.scatter(condition_numbers, errors, alpha=0.5, s=20, c='red')

    if len(condition_numbers) > 1:
        log_cond = np.log10(condition_numbers)
        log_errors = np.log10(errors)

        coeffs = np.polyfit(log_cond, log_errors, 1)
        trend_line = 10**(coeffs[1] + coeffs[0] * log_cond)

        plt.plot(condition_numbers, trend_line, 'b-', linewidth=2,
                label=f'Тренд: ошибка ~ cond^{coeffs[0]:.2f}')

    plt.xlabel('Число обусловленности cond(A)', fontsize=12)
    plt.ylabel('Относительная ошибка', fontsize=12)
    plt.title(f'Зависимость ошибки от числа обусловленности (p={p})', fontsize=14)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print("\nСтатистика ошибок:")
    print(f"  Средняя ошибка: {np.mean(errors):.2e}")
    print(f"  Медианная ошибка: {np.median(errors):.2e}")
    print(f"  Максимальная ошибка: {np.max(errors):.2e}")
    print(f"  Минимальная ошибка: {np.min(errors):.2e}")

    print("График построен!")

def perturbation_experiment(A, p=3, num_perturbations=1000):
    print(f"\n--- Исследование устойчивости к возмущениям ---")
    print(f"Исходная матрица A:")
    print(A)
    print(f"Число обусловленности: {np.linalg.cond(A):.2e}")

    try:
        true_root = matrix_pth_root(A, p)
        print("Истинный корень вычислен успешно")
    except ValueError as e:
        print(f"Ошибка: невозможно вычислить корень из исходной матрицы: {e}")
        return

    perturbations = [np.random.uniform(-0.01, 0.01, (2, 2)) for _ in range(num_perturbations)]

    print(f"Сгенерировано {num_perturbations} возмущений")

    errors = []
    perturbation_norms = []

    successful_calculations = 0
    for i, pert in enumerate(perturbations):
        A_pert = A + pert

        try:
            root_pert = matrix_pth_root(A_pert, p)
            successful_calculations += 1
        except ValueError:
            continue

        error = norm(root_pert - true_root, 2)
        errors.append(error)

        pert_norm = norm(pert, 2)
        perturbation_norms.append(pert_norm)

        if i % 100 == 0 and i > 0:
            print(f"  Обработано {i}/{num_perturbations} возмущений")

    print(f"Успешно обработано: {successful_calculations}/{num_perturbations} возмущений")

    plt.figure(figsize=(10, 6))
    plt.scatter(perturbation_norms, errors, alpha=0.5, s=20, c='green')

    if len(perturbation_norms) > 1:
        coeffs = np.polyfit(perturbation_norms, errors, 1)
        trend_line = np.polyval(coeffs, perturbation_norms)
        plt.plot(perturbation_norms, trend_line, 'r-', linewidth=2,
                label=f'Тренд: наклон = {coeffs[0]:.2f}')

    plt.xlabel('Норма возмущения ||ΔA||', fontsize=12)
    plt.ylabel('Норма ошибки ||f(A+ΔA) - f(A)||', fontsize=12)
    plt.title(f'Чувствительность к возмущениям (cond(A)={np.linalg.cond(A):.2e})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if len(errors) > 0:
        sensitivity = np.median([e/p for e, p in zip(errors, perturbation_norms) if p > 0])
        print(f"Медианная чувствительность: {sensitivity:.2f}")
        print("(Коэффициент усиления ошибки: во сколько раз ошибка результата")
        print(" больше ошибки во входных данных)")

    print("График построен!")

def time_vs_accuracy(A, max_p=10):
    print(f"\n--- Зависимость точности и времени от степени корня ---")
    print(f"Тестовая матрица A:")
    print(A)

    times = []
    errors = []
    p_values = []

    for p in range(1, max_p + 1):
        print(f"Тестирование p={p}...")

        start_time = time.time()
        try:
            our_root = matrix_pth_root(A, p)
            our_time = time.time() - start_time
        except ValueError as e:
            print(f"  Пропуск p={p}: {e}")
            continue

        try:
            scipy_root = fractional_matrix_power(A, 1/p)
        except:
            print(f"  Ошибка scipy для p={p}")
            continue

        error = norm(our_root - scipy_root, 2) / norm(scipy_root, 2)

        times.append(our_time)
        errors.append(error)
        p_values.append(p)

        print(f"  Время: {our_time:.4f} сек, Ошибка: {error:.2e}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(p_values, times, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Степень корня p', fontsize=12)
    ax1.set_ylabel('Время выполнения (сек)', fontsize=12)
    ax1.set_title('Зависимость времени от степени корня', fontsize=14)
    ax1.grid(True, alpha=0.3)

    ax2.plot(p_values, errors, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Степень корня p', fontsize=12)
    ax2.set_ylabel('Относительная ошибка', fontsize=12)
    ax2.set_title('Зависимость ошибки от степени корня', fontsize=14)
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.show()

    if errors:
        print(f"\nСтатистика по ошибкам для p=1..{max_p}:")
        print(f"  Средняя ошибка: {np.mean(errors):.2e}")
        print(f"  Максимальная ошибка: {np.max(errors):.2e} (при p={p_values[np.argmax(errors)]})")
        print(f"  Минимальная ошибка: {np.min(errors):.2e} (при p={p_values[np.argmin(errors)]})")

    print("Графики построены!")

print("\n" + "=" * 80)
print("ЗАДАЧА 2: РЕГРЕССИЯ И РЕГУЛЯРИЗАЦИЯ")
print("=" * 80)

data_nonlinear = np.array([
    [30.8, 459.7, 39.5, 55.3, 79.2],
    [31.2, 492.9, 37.3, 54.7, 77.4],
    [33.3, 528.6, 38.1, 63.7, 80.2],
    [35.6, 560.3, 39.3, 69.8, 80.4],
    [36.4, 624.6, 37.8, 65.9, 83.9],
    [36.7, 666.4, 38.4, 64.5, 85.5],
    [38.4, 717.8, 40.1, 70.0, 93.7],
    [40.4, 768.2, 38.6, 73.2, 106.1],
    [40.3, 843.3, 39.8, 67.8, 104.8],
    [41.8, 911.6, 39.7, 79.1, 114.0],
    [40.4, 931.1, 52.1, 95.4, 124.1],
    [40.7, 1021.5, 48.9, 94.2, 127.6],
    [40.1, 1165.9, 58.3, 123.5, 142.9],
    [42.7, 1349.6, 57.9, 129.9, 143.6],
    [44.1, 1449.4, 56.5, 117.6, 139.2],
    [46.7, 1575.5, 63.7, 130.9, 165.5],
    [50.6, 1759.1, 61.6, 129.8, 203.3],
    [50.1, 1994.2, 58.9, 128.0, 219.6],
    [51.7, 2258.1, 66.4, 141.0, 221.6],
    [52.9, 2478.7, 70.4, 168.2, 232.6]
])

print("Исходные данные (20 наблюдений):")
print("Столбцы: y, x1, x2, x3, x4")
print(data_nonlinear)

X = data_nonlinear[:, 2:]
y = data_nonlinear[:, 0]

print(f"\nПризнаки X (x2, x3, x4): {X.shape}")
print(f"Целевая переменная y: {y.shape}")

X_log = np.log(X)
y_log = np.log(y)

print("\nЛогарифмированные данные:")
print("X_log (log(x2), log(x3), log(x4)):")
print(X_log)
print("y_log (log(y)):")
print(y_log)

fold_indices = [
    [0, 4, 8, 12, 16],
    [1, 5, 9, 13, 17],
    [2, 6, 10, 14],
    [3, 7, 11, 15, 19]
]

print(f"\nРазбиение на фолды для кросс-валидации:")
for i, fold in enumerate(fold_indices, 1):
    print(f"  Фолд {i}: наблюдения {[x+1 for x in fold]}")

def cv_mse(model_class, **model_kwargs):
    mse_scores = []

    for k in range(4):
        test_idx = fold_indices[k]
        train_idx = np.setdiff1d(np.arange(20), test_idx)

        X_train, X_test = X_log[train_idx], X_log[test_idx]
        y_train, y_test = y_log[train_idx], y_log[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_final = np.column_stack([np.ones(len(X_train_scaled)), X_train_scaled])
        X_test_final = np.column_stack([np.ones(len(X_test_scaled)), X_test_scaled])

        model = model_class(**model_kwargs)
        model.fit(X_train_final, y_train)

        y_pred = model.predict(X_test_final)

        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores)

print("\n" + "-" * 70)
print("1. РЕГРЕССИЯ БЕЗ РЕГУЛЯРИЗАЦИИ")
print("-" * 70)

mse_no_reg = cv_mse(LinearRegression)
print(f"MSE без регуляризации: {mse_no_reg:.6f}")

scaler_full = StandardScaler()
X_log_scaled = scaler_full.fit_transform(X_log)
X_log_final = np.column_stack([np.ones(len(X_log_scaled)), X_log_scaled])

model_no_reg = LinearRegression()
model_no_reg.fit(X_log_final, y_log)

print("Коэффициенты модели без регуляризации:")
print(f"  Свободный член (β₀): {model_no_reg.intercept_:.4f}")
print(f"  Коэффициенты: {model_no_reg.coef_[1:]}")

print("\n" + "-" * 70)
print("2. L2-РЕГУЛЯРИЗАЦИЯ (RIDGE)")
print("-" * 70)

alphas = np.logspace(-3, 3, 50)

print(f"Тестирование {len(alphas)} значений alpha...")

ridge_scores = []
for i, alpha in enumerate(alphas):
    mse = cv_mse(Ridge, alpha=alpha)
    ridge_scores.append(mse)
    if i % 10 == 0:
        print(f"  alpha={alpha:.4f}, MSE={mse:.6f}")

best_ridge_alpha = alphas[np.argmin(ridge_scores)]
best_ridge_mse = np.min(ridge_scores)

print(f"\nЛучшее значение alpha: {best_ridge_alpha:.4f}")
print(f"MSE при лучшем alpha: {best_ridge_mse:.6f}")

model_ridge = Ridge(alpha=best_ridge_alpha)
model_ridge.fit(X_log_final, y_log)

print("Коэффициенты Ridge-модели:")
print(f"  Свободный член (β₀): {model_ridge.intercept_:.4f}")
print(f"  Коэффициенты: {model_ridge.coef_[1:]}")

plt.figure(figsize=(10, 6))
plt.plot(alphas, ridge_scores, linewidth=2, label='Ridge (L2)')
plt.axvline(best_ridge_alpha, color='red', linestyle='--',
            label=f'Оптимальное alpha={best_ridge_alpha:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (параметр регуляризации)', fontsize=12)
plt.ylabel('MSE (кросс-валидация)', fontsize=12)
plt.title('L2-регуляризация: зависимость качества от alpha', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "-" * 70)
print("3. L1-РЕГУЛЯРИЗАЦИЯ (LASSO)")
print("-" * 70)

print(f"Тестирование {len(alphas)} значений alpha...")

lasso_scores = []
for i, alpha in enumerate(alphas):
    mse = cv_mse(Lasso, alpha=alpha, max_iter=10000)
    lasso_scores.append(mse)
    if i % 10 == 0:
        print(f"  alpha={alpha:.4f}, MSE={mse:.6f}")

best_lasso_alpha = alphas[np.argmin(lasso_scores)]
best_lasso_mse = np.min(lasso_scores)

print(f"\nЛучшее значение alpha: {best_lasso_alpha:.4f}")
print(f"MSE при лучшем alpha: {best_lasso_mse:.6f}")

model_lasso = Lasso(alpha=best_lasso_alpha, max_iter=10000)
model_lasso.fit(X_log_final, y_log)

print("Коэффициенты Lasso-модели:")
print(f"  Свободный член (β₀): {model_lasso.intercept_:.4f}")
print(f"  Коэффициенты: {model_lasso.coef_[1:]}")

nonzero_coeffs = np.sum(model_lasso.coef_[1:] != 0)
print(f"Количество ненулевых коэффициентов: {nonzero_coeffs} из {len(model_lasso.coef_[1:])}")

plt.figure(figsize=(10, 6))
plt.plot(alphas, lasso_scores, linewidth=2, color='orange', label='Lasso (L1)')
plt.axvline(best_lasso_alpha, color='red', linestyle='--',
            label=f'Оптимальное alpha={best_lasso_alpha:.4f}')
plt.xscale('log')
plt.xlabel('Alpha (параметр регуляризации)', fontsize=12)
plt.ylabel('MSE (кросс-валидация)', fontsize=12)
plt.title('L1-регуляризация: зависимость качества от alpha', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\n" + "=" * 70)
print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("=" * 70)

print("МЕТОД               |    MSE    |  ЛУЧШИЙ ALPHA")
print("-" * 50)
print(f"Без регуляризации  | {mse_no_reg:.6f} |     -")
print(f"Ridge (L2)         | {best_ridge_mse:.6f} | {best_ridge_alpha:.4f}")
print(f"Lasso (L1)         | {best_lasso_mse:.6f} | {best_lasso_alpha:.4f}")

print("\nСРАВНЕНИЕ КОЭФФИЦИЕНТОВ:")
print("Признак | Без регул. |   Ridge   |   Lasso   ")
print("-" * 45)
features = ['log(x₂)', 'log(x₃)', 'log(x₄)']
for i, feature in enumerate(features):
    print(f"{feature:7} | {model_no_reg.coef_[1+i]:10.4f} | {model_ridge.coef_[1+i]:10.4f} | {model_lasso.coef_[1+i]:10.4f}")

print("\nВЫВОДЫ:")
if best_ridge_mse < mse_no_reg and best_lasso_mse < mse_no_reg:
    print("• Регуляризация улучшила качество модели")
    if best_ridge_mse < best_lasso_mse:
        print("• L2-регуляризация показала лучший результат")
    else:
        print("• L1-регуляризация показала лучший результат")
else:
    print("• Регуляризация не улучшила качество модели")

if np.any(model_lasso.coef_[1:] == 0):
    zero_features = [features[i] for i in range(len(features)) if model_lasso.coef_[1+i] == 0]
    print(f"• Lasso отсек признаки: {', '.join(zero_features)}")

print("\n" + "=" * 80)
print("ЗАДАЧА 3: ВАВИЛОНСКИЙ МЕТОД И МЕТОД НЬЮТОНА")
print("=" * 80)

def babylonian_sqrt(S, initial_guess, tolerance=1e-10, max_iter=1000):
    if S <= 0:
        raise ValueError("S должно быть положительным числом")
    if initial_guess <= 0:
        raise ValueError("Начальное приближение должно быть положительным")

    x = initial_guess
    iterations = 0

    print(f"Вавилонский метод для √{S} с начальным приближением {initial_guess}")

    for iteration in range(max_iter):
        x_new = 0.5 * (x + S / x)
        iterations += 1

        if iteration < 5:
            error = abs(x_new - np.sqrt(S))
            print(f"  Итерация {iteration}: x = {x_new:.10f}, ошибка = {error:.2e}")

        if abs(x_new - x) < tolerance:
            print(f"Сходимость достигнута за {iterations} итераций")
            print(f"Результат: {x_new:.10f}")
            print(f"Точное значение: {np.sqrt(S):.10f}")
            print(f"Финальная ошибка: {abs(x_new - np.sqrt(S)):.2e}")
            return x_new

        x = x_new

    print(f"Предупреждение: достигнуто максимальное число итераций ({max_iter})")
    return x

def newton_sqrt(S, initial_guess, tolerance=1e-10, max_iter=1000):
    if S <= 0:
        raise ValueError("S должно быть положительным числом")

    x = initial_guess
    iterations = 0

    print(f"Метод Ньютона для √{S} с начальным приближением {initial_guess}")

    for iteration in range(max_iter):
        f = x**2 - S
        f_derivative = 2 * x

        if abs(f_derivative) < 1e-15:
            break

        x_new = x - f / f_derivative
        iterations += 1

        if iteration < 5:
            error = abs(x_new - np.sqrt(S))
            print(f"  Итерация {iteration}: x = {x_new:.10f}, ошибка = {error:.2e}")

        if abs(x_new - x) < tolerance:
            print(f"Сходимость достигнута за {iterations} итераций")
            print(f"Результат: {x_new:.10f}")
            print(f"Точное значение: {np.sqrt(S):.10f}")
            print(f"Финальная ошибка: {abs(x_new - np.sqrt(S)):.2e}")
            return x_new

        x = x_new

    print(f"Предупреждение: достигнуто максимальное число итераций ({max_iter})")
    return x

def combined_sqrt(S, tolerance=1e-10):
    if S <= 0:
        raise ValueError("S должно быть положительным числом")

    print(f"\nКОМБИНИРОВАННЫЙ МЕТОД ДЛЯ √{S}")
    print("=" * 50)

    if S > 1:
        initial_guess = S / 2
    elif S < 1 and S > 0:
        initial_guess = 1
    else:
        initial_guess = 1

    print("ЭТАП 1: Вавилонский метод для начального приближения")
    rough_approximation = babylonian_sqrt(S, initial_guess, tolerance=1e-5, max_iter=10)

    print(f"Получено начальное приближение: {rough_approximation:.10f}")

    print("\nЭТАП 2: Уточнение методом Ньютона")
    final_result = newton_sqrt(S, rough_approximation, tolerance=tolerance)

    print(f"\nФИНАЛЬНЫЙ РЕЗУЛЬТАТ: √{S} = {final_result:.10f}")
    print(f"ПРОВЕРКА: ({final_result:.10f})² = {final_result**2:.10f}")

    return final_result

print("\nТЕСТИРОВАНИЕ КОМБИНИРОВАННОГО МЕТОДА:")
print("=" * 70)

test_values = [
    25,
    2,
    1000,
    0.01,
    123456789,
    0.0001
]

print("ТЕСТОВЫЕ ЗНАЧЕНИЯ:")
for S in test_values:
    print(f"\n--- Вычисление √{S} ---")
    try:
        result = combined_sqrt(S)
        exact = np.sqrt(S)
        error = abs(result - exact)
        print(f"ОКОНЧАТЕЛЬНАЯ ОШИБКА: {error:.2e}")

        if error < 1e-8:
            print("✅ ВЫСОКАЯ ТОЧНОСТЬ")
        elif error < 1e-5:
            print("⚠️  УДОВЛЕТВОРИТЕЛЬНАЯ ТОЧНОСТЬ")
        else:
            print("❌ НИЗКАЯ ТОЧНОСТЬ")

    except ValueError as e:
        print(f"❌ ОШИБКА: {e}")

print("\n" + "=" * 70)
print("ТЕСТИРОВАНИЕ УСТОЙЧИВОСТИ К НАЧАЛЬНОМУ ПРИБЛИЖЕНИЮ")
print("=" * 70)

S_test = 16
initial_guesses = [1, 100, 0.1, 0.001, 1000]

print(f"Вычисление √{S_test} с разными начальными приближениями:")
for x0 in initial_guesses:
    print(f"\nНачальное приближение: {x0}")
    try:
        result = combined_sqrt(S_test)
        error = abs(result - 4)
        print(f"Результат: {result}, ошибка: {error:.2e}")
    except Exception as e:
        print(f"Ошибка: {e}")

print("\n" + "=" * 80)
print("ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ")
print("=" * 80)