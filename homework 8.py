from random import seed
import numpy as np

seed(2024)
np.random.seed(2024)

# Данные для x и y
x = np.array([1, 5, 1, 1, 12, 8])  # ваше распределение для x_
y = np.array([2, 9, 3, 4, 10, 14])  # ваше распределение для Y_

# Начальная инициализация весов
coef_i = 1.0
coef_5 = 1.0

# Гиперпараметры
learning_rate = 0.01
n_iterations = 10000

# Функция для вычисления RMSE
def compute_rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))

# Градиентный спуск
for iteration in range(n_iterations):
    # Прогнозы
    y_pred = coef_i * x + coef_5 * 5
    # Ошибка
    error = y_pred - y
    
    # Градиенты
    gradient_i = 2 * np.mean(error * x)
    gradient_5 = 2 * np.mean(error * 5)
    
    # Обновление весов
    coef_i -= learning_rate * gradient_i
    coef_5 -= learning_rate * gradient_5
    
    # Вычисляем RMSE каждые 1000 итераций
    if iteration % 1000 == 0:
        rmse = compute_rmse(y_pred, y)
        print(f"Iteration {iteration}, RMSE: {rmse}")

# Итоговые коэффициенты и RMSE
y_final_pred = coef_i * x + coef_5 * 5
final_rmse = compute_rmse(y_final_pred, y)

print(f"Final coef_i: {coef_i}, Final coef_5: {coef_5}, Final RMSE: {final_rmse}")
