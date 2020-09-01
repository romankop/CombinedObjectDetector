# Импорт библиотек
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import csv

# Словарь из доступных в open-cv трекеров

TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create,
    "goturn": cv2.TrackerGOTURN_create
    # Для goturn требуется установить goturn.prototxt и goturn.caffemodel
}

# Массив из меток классов в моделях детекции в torchvisions

with open('labels.txt') as labels:
    LABELS = np.array(list(csv.reader(labels))[0])


class Detector:
    """
    Класс, выполняющий функции детектора.
    Принимает на вход модель детектора, которая будут
    использоваться классом для обработки кадров.
    Параметры
    ----------
    model : torch.nn.Module
          Модель детекции из библиотеки torchvision.models или любая
          модель, обладающая тем же функционалом.
    USE_CUDA : bool, default False, optional
          Флаг, указывающий, нужно ли переводить использование модели
          на GPU.
    """

    def __init__(self, model: torch.nn.Module, USE_CUDA: bool = False):
        """ Метод инициализации класcа """

        self.detector = model

        torch.set_grad_enabled(False)
        self.detector.eval()

        if torch.cuda.is_available() and USE_CUDA:
            self.device = torch.device("cuda")
            self.detector.cuda()
        else:
            self.device = torch.device("cpu")

    def transform(self, image: np.ndarray) -> torch.Tensor:

        """
        Функция для форматирования картинки под torch-модель.
        Производит трансформацию данных о картинке.
        Параметры
        ----------
        image : np.ndarray
            Картинка в формате np.ndarray размерности (H, W, C).
        Результат
        ----------
        torch.Tensor
            Отформатированная картинка в формате torch.Tensor.
        """

        transform_pipe = T.Compose([T.ToTensor()])

        return transform_pipe(image)

    def trim(self, boxes: np.ndarray, format: tuple) -> np.ndarray:

        """
        Функция, вписывающая найденный бокс в рамки картинки.
        Обрезает боксы, которые выходят за рамки картинки, по её границам.
        Параметры
        ----------
        boxes : np.ndarray
            Массив боксов, представляющих из себя array
            формата Ndarray[x1, y1, x2, y2].
        format : tuple
            Разрешение картинки в виде кортежа из трех элементов:
            количество пикселей по вертикали и горизонтали,
            количество каналов.
        Результат
        ----------
        np.ndarray
            Массив обработанных боксов.
        """

        boxes = np.maximum(0, boxes)

        boxes[:, (0, 2)] = np.minimum(format[1] - 1, boxes[:, (0, 2)])
        boxes[:, (1, 3)] = np.minimum(format[0] - 1, boxes[:, (1, 3)])

        return boxes

    def __call__(self, image: np.ndarray) -> tuple:

        """
        Метод вызова функции.
        Производит предсказание объектов и их положения с помощью детектора.
        Параметры
        ----------
        image : np.ndarray
            Картинка в формате np.ndarray размерности (H, W, C).
        Результат
        ----------
        Tuple
            Кортеж из боксов, идентификаторов объектов на боксах и
            вероятностей их нахождения.
        """

        results = ['boxes', 'labels', 'scores']
        image_tensor = self.transform(image).to(self.device)

        detections = self.detector([image_tensor])

        boxes_detector, label_id, scores = map(lambda x:
                                               detections[0][x].cpu().numpy(),
                                               results)

        boxes_detector = self.trim(boxes_detector, image.shape)

        return boxes_detector, label_id, scores


class Tracker:
    """
    Класс, выполняющий функции трекера.
    Принимает на вход название модели трекера, представленной в opencv-python,
    которая будут использоваться классом для обработки кадров.
    Параметры
    ----------
    tracker_type : {'csrt', 'kcf', 'boosting', 'mil', 'tld', 'medianflow',
                    'mosse', 'goturn'}, default None
          Строка, являющаяся ключом к словарю моделей трекеров TRACKERS.
    """

    def __init__(self, tracker_type: str):
        """ Метод инициализации класcа """

        self.tracker = TRACKERS[tracker_type]
        self.trackers = cv2.MultiTracker_create()

        self.current_image = None

    def __call__(self, image: np.ndarray) -> tuple:

        """
        Метод вызова функции.
        Производит предсказание положения объектов с помощью трекера.
        Параметры
        ----------
        image : np.ndarray
            Картинка в формате np.ndarray размерности (H, W, C).
        Результат
        ----------
        np.ndarray or Tuple
            Список обнаруженных боксов в формате np.ndarray, если они
            обнаружены, и пустой кортеж в обратной ситуации.
        """

        self.current_image = image

        _, boxes_tracker = self.trackers.update(image)

        return boxes_tracker

    def update(self, boxes):

        self.trackers = cv2.MultiTracker_create()

        for box in boxes:
            box = tuple(box)
            self.trackers.add(self.tracker(), self.current_image, box)

        return
