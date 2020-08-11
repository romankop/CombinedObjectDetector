# Импорт библиотек
import numpy as np


# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh: float = 0.7):
    """
    Функция для подавления схожих боксов.

    Функция принимает набор боксов и возвращает неперекрывающиеся боксы,
    используя Non-Max Suppression Algorithm.

    Параметры
    ----------
    boxes: np.ndarray
        Двумерный array, содержащий в себе N списков
        формата Ndarray[x1, y1, x2, y2].

    overlapThresh: float, default 0.7
        Пороговое значение для Non-Maximum Suppression Algorithm.

    Результат
    ----------
    Ndarray[Ndarray[x1, y1, x2, y2]]
        Набор непересекающихся друг с другом боксов.

    """
    # Возврат при отсутствии входных данных
    if len(boxes) == 0:
        return []

    # Приведение к типу данных float
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # Создаем список выбранных индексов боксов
    pick = []

    # Извлечение координат боксов
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Считаем площадь боксов и сортируем по правому верхнему углу
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # Итерация по оставшимся элементам в списке индексов
    while len(idxs) > 0:
        # Взятие бокса по последнему индексу в списке
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Нахождение координат пересечений с выбранным боксом
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Вычисление широты и высоты боксов
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Нахождение индекса перекрытия
        overlap = (w * h) / area[idxs[:last]]

        # Удаление индексов боксов, сильно пересекающихся с выбранным
        idxs = np.delete(idxs,
                         np.concatenate(([last],
                                        np.where(overlap > overlapThresh)[0])))

    # Возвращение нужных боксов
    return boxes[pick]
