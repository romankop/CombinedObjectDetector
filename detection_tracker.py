# Импорт библиотек
import numpy as np
import cv2
import torch
import torchvision.transforms as T
import csv
from nms import non_max_suppression_fast as nms

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


class DetectionTracker:
    """
    Класс, выполняющий функции как трекера, так и детектора.

    Принимает на вход модели трекера и детектора, которые будут
    использоваться классом для обработки кадров.

    Параметры
    ----------
    detect : torch.nn.Module
          Модель детекции из библиотеки torchvision.models или любая
          модель, обладающая тем же функционалом.

    tracker_type : {'csrt', 'kcf', 'boosting', 'mil', 'tld', 'medianflow',
                    'mosse', 'goturn'}, default None
          Строка, являющаяся ключом к словарю моделей трекеров TRACKERS.

    detection_decim : int, default 0, optional
          Количество пропускаемых для детекции кадров.

    score_threshold : float, default 0.7, optional
          Пороговая вероятность для выбора моделью детекции объектов.

    overlap_threshold : float, default 0.7, optional
          Пороговое значение для Non-Maximum Suppression Algorithm.

    USE_CUDA : bool, default False, optional
          Флаг, указывающий, нужно ли переводить использование модели
          на GPU.
    """

    def __init__(self, detector: torch.nn.Module, tracker_type=None,
                 detection_decim: int = 0, score_threshold: float = 0.7,
                 overlap_threshold: float = 0.7, USE_CUDA: bool = False):
        """ Метод инициализации класcа """

        self.detector = Detector(model=detector,
                                 USE_CUDA=USE_CUDA)

        if tracker_type is not None and tracker_type not in TRACKERS:
            raise ValueError('Tracker is not represented in Open-CV!')

        if tracker_type is None and detection_decim > 0:
            raise ValueError("Module won't be able to perform detection \
                              on some frames!")

        self.tracker = Tracker(tracker_type) if tracker_type is not None \
            else tracker_type
        self.decim = detection_decim
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold

        self.iter = 0

    def _iou(self, box: np.ndarray, boxes: np.ndarray) -> int:
        """
        Функция для нахождения наиболее похожего бокса к изначальному.

        Функция принимает бокс и набор потенциально похожих на него боксов,
        возвращая индекс наиболее схожего бокса из представленного набора.

        Параметры
        ----------
        box : np.ndarray
            Бокс, представляющий из себя array формата Ndarray[x1, y1, x2, y2].

        boxes: np.ndarray
            Двумерный array, содержащий в себе N списков того же формата,
            что и бокс.

        Результат
        ----------
        int
            Номер наиболее похожего бокса в наборе.

        """

        # Вычисление площадей боксов
        box_square = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
        boxes_square = (boxes[:, 2] - boxes[:, 0] + 1) * \
                       (boxes[:, 3] - boxes[:, 1] + 1)

        # Нахождение координат пересечений боксов
        inter_x1 = np.maximum(box[0], boxes[:, 0])
        inter_y1 = np.maximum(box[1], boxes[:, 1])
        inter_x2 = np.minimum(box[2], boxes[:, 2])
        inter_y2 = np.minimum(box[3], boxes[:, 3])

        # Вычисление площадей внутренних пересечений
        inter_area = (inter_x2 - inter_x1 + 1) * (inter_y2 - inter_y1 + 1)

        # Вычисление метрики IoU
        iou_s = inter_area / (box_square + boxes_square - inter_area) * \
            ((inter_x2 - inter_x1 >= 0) & (inter_y2 - inter_y1 >= 0))

        return np.argmax(iou_s)

    def __call__(self, image: np.ndarray) -> list:

        """
        Метод вызова функции.

        Производит предсказание объектов и их положения с помощью трекера
        и детектора.

        Параметры
        ----------
        image : np.ndarray
            Картинка в формате np.ndarray размерности (H, W, C).

        Результат
        ----------
        List[Dict]
            Список из словарей, которые характеризуют найденные объекты.
            В словарях информация о расположении верхнего левого угла,
            высоте, ширине, классе объекта и вероятности принадлежности
            классу.

        """

        # Определение необходимости использования детектора
        det_status = (self.decim == 0) or \
            (self.iter % self.decim == 0)

        # Увеличиваем счетчик
        self.iter += 1

        # Получение результатов из детектора
        if det_status:

            boxes_detector, label_id, scores = self.detector(image)
            labels = LABELS[label_id]

            mask = scores > self.score_threshold

            boxes_detector = boxes_detector[mask]
            scores = np.atleast_1d(scores[mask])
            labels = np.atleast_1d(labels[mask])

        # Получение результатов из трекера и формирование общих результатов
        if self.tracker is not None:

            boxes_tracker = self.tracker(image)

            if det_status and scores.size:

                if isinstance(boxes_tracker, np.ndarray):

                    boxes_inter = np.concatenate((boxes_tracker,
                                                  boxes_detector))
                else:

                    boxes_inter = boxes_detector

                boxes_inter = nms(boxes_inter, self.overlap_threshold)

            else:

                if isinstance(boxes_tracker, np.ndarray):

                    boxes_inter = nms(boxes_tracker, self.overlap_threshold)

                    # Если детектор не работает, берём предыдущие данные
                    boxes_detector, labels, scores = self.prev_objects

                else:

                    boxes_inter = np.array([])

            # Обновляем трекер и добавляем в него объекты

            self.tracker.update(boxes_inter)

        elif boxes_detector.size:

            boxes_inter = nms(boxes_detector, self.overlap_threshold)

        else:

            boxes_inter = np.array([])

        # Если боксов нет, то возвращаем пустой массив объектов
        if not boxes_inter.size:

            return []

        # Определяем, каким объектам соответствуют боксы
        indexes_to_choose = np.apply_along_axis(self._iou, 1,
                                                boxes_inter,
                                                boxes=boxes_detector
                                                ).astype(int)

        boxes_detector_chosen = boxes_detector[indexes_to_choose, :]
        labels_chosen = labels[indexes_to_choose]
        scores_chosen = scores[indexes_to_choose]

        # Вычисляем признаки объекта
        height = boxes_detector_chosen[:, 3] - boxes_detector_chosen[:, 1]
        width = boxes_detector_chosen[:, 2] - boxes_detector_chosen[:, 0]

        up_left_corner = boxes_detector_chosen[:, (0, 3)]
        up_left_corner[:,  1] = up_left_corner[:, 1] - height

        # Формируем конечный результат
        objects = []
        for i in range(height.shape[0]):
            _object = {'up_left_corner': up_left_corner[i, :],
                       'height': height[i],
                       'width': width[i],
                       'score': scores_chosen[i],
                       'label': labels_chosen[i]}

            objects.append(_object)

        # Сохраняем данные с текущей итерации
        self.prev_objects = (boxes_detector_chosen,
                             labels_chosen, scores_chosen)

        return objects
