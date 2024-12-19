import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from scipy.spatial import Voronoi, voronoi_plot_2d

# Назва файлу
file_name = "DS5.txt"

# Функція для зчитування датасету з файлу
def read_dataset(file_name):
    try:
        with open(file_name, "r") as file:
            data = [line.strip().split() for line in file]
            coordinates = [(int(x), int(y)) for x, y in data]
            return np.array(coordinates)
    except FileNotFoundError:
        print(f"Файл {file_name} не знайдено.")
        return []
    except ValueError:
        print("Помилка у форматі даних. Перевірте файл.")
        return []

# Зчитування координат
coordinates = read_dataset(file_name)

if coordinates.size > 0:
    # Кластеризація за допомогою DBSCAN
    dbscan = DBSCAN(eps=10, min_samples=5)  # Параметри DBSCAN
    labels = dbscan.fit_predict(coordinates)

    # Знаходимо центри ваги для кожного кластеру
    unique_labels = set(labels)
    centroids = []
    for label in unique_labels:
        if label != -1:  # Пропускаємо шуми
            cluster_points = coordinates[labels == label]
            centroid = np.mean(cluster_points, axis=0)
            centroids.append(centroid)

    centroids = np.array(centroids)

    # Побудова діаграми Вороного для центрів ваги
    vor = Voronoi(centroids)

    # Побудова графіка
    plt.figure(figsize=(9.6, 5.4))  # Встановлення розміру полотна
    plt.scatter(coordinates[:, 0], coordinates[:, 1], color='black', alpha=0.1, label="Точки датасету")
    plt.scatter(centroids[:, 0], centroids[:, 1], color='red', s=25, label="Центри ваги")

    # Відображення діаграми Вороного
    voronoi_plot_2d(vor, ax=plt.gca(), show_vertices=False, line_colors='blue', line_width=1)

    # Налаштування осей
    plt.xlim(0, np.max(coordinates[:, 0]) * 1.1)
    plt.ylim(0, np.max(coordinates[:, 1]) * 1.1)
    plt.gca().set_aspect('equal', adjustable='box')

    # Підписи і легенда
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Точки, Центри Ваги та Діаграма Вороного")
    plt.legend()

    # Збереження результату у графічний файл
    output_file = "voronoi_with_centroids.png"
    plt.savefig(output_file, dpi=100)
    plt.close()

    print(f"Результат збережено у файл: {output_file}")
else:
    print("Не вдалося зчитати дані.")
