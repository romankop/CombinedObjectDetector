# Импорт библиотек
import numpy as np
import cv2
import torch
import torchvision
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd
from nms import non_max_suppression_fast as nms

import warnings
warnings.filterwarnings("ignore")

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
}

# Массив из меток классов в моделях детекции в torchvisions

LABELS = np.array([
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack',
    'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A',
    'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'N/A', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 
    'toothbrush'
])

class DetectionTracker:
    """
    Класс, выполняющий функции как трекера, так и детектора.

    Принимает на вход модели трекера и детектора, которые будут
    использоваться классом для обработки кадров.

    Параметры
    ----------
    detect : torchvision.models.detection
          Модель детекции из библиотеки torchvision.models или любая
          модель, обладающая тем же функционалом.

    tracker : {'csrt', 'kcf', 'boosting', 'mil', 'tld', 'medianflow',
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
    
    def __init__(self, detector: torchvision.models.detection, tracker=None,
                 detection_decim: int = 0, score_threshold: float = 0.7, 
                 overlap_threshold: float = 0.7, USE_CUDA: bool = False):
        """Метод инициализации класcа"""
                
        self.detector = detector

        if tracker is not None and tracker not in TRACKERS:
            raise ValueError('Tracker is not represented in Open-CV!')

        if tracker is None and detection_decim > 0:
            raise ValueError("Modulу won't be able to perform detection \
                              on some frames!")
        
        self.tracker = TRACKERS[tracker] if tracker is not None else tracker
        self.decim = detection_decim
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold

        if torch.cuda.is_available() and USE_CUDA:
              self.device = torch.device("cuda")
              self.detector.cuda()
        else:
              self.device = torch.device("cpu")
        
        self.trackers = cv2.MultiTracker_create()
        self.iter = 0

    def _iou(self, box: np.ndarray, boxes: np.ndarray,
             eps: float = 1e-5) -> int:
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

        eps: float, default 1e-5, optional
            Поправочная надбавка, чтобы не обнулялся знаменатель в вычислении
            метрики IoU.

        Результат
        ----------
        int 
            Номер наиболее похожего бокса в наборе.

        """

        # Вычисление площадей боксов
        box_square = (box[2] - box[0]) * (box[3] - box[1])
        boxes_square = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        # Нахождение координат пересечений боксов
        inter_x1 = np.maximum(box[0], boxes[:, 0])
        inter_y1 = np.maximum(box[1], boxes[:, 1])
        inter_x2 = np.minimum(box[2], boxes[:, 2])
        inter_y2 = np.minimum(box[3], boxes[:, 3])

        # Вычисление площадей внутренних пересечений
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

        # Вычисление метрики IoU
        iou_s = inter_area / (box_square + boxes_square - inter_area + eps) * \
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
        det_status = (self.decim == 0) or (self.iter % self.decim == 0)
        
        # Получение результатов из детектора
        if det_status:

          transform = T.Compose([T.ToTensor()])

          with torch.no_grad():

              image_tensor = transform(image).to(self.device)
              
              model.eval()

              detections = self.detector([image_tensor])

              boxes_detector, label_idx, scores = map(lambda x: x.cpu(),
                                                      detections[0].values())
              labels = LABELS[label_idx]

              mask = scores > self.score_threshold

              scores = scores[mask].numpy()
              boxes_detector = boxes_detector[mask].numpy()
              labels = labels[mask]

        # Получение результатов из трекера и формирование общих результатов
        if self.tracker is not None:
            
            _, boxes_tracker = self.trackers.update(image)

            if det_status:

                if isinstance(boxes_tracker, np.ndarray):
                    
                    boxes_inter = np.concatenate((boxes_tracker, 
                                                  boxes_detector))
                else:

                    boxes_inter = boxes_detector
                
                boxes_inter = nms(boxes_inter, self.overlap_threshold)
            
            else:

                boxes_inter = nms(boxes_tracker, self.overlap_threshold)

                # Если детектор не используется, используем предыдущие данные
                boxes_detector, label_idx, scores = self.prev_objects
                labels = LABELS[label_idx]
        
        else:

            boxes_inter = nms(boxes_detector, self.overlap_threshold)

        # Создаем новый трекер и добавляем в него объекты
        self.trackers = cv2.MultiTracker_create()

        for box in boxes_inter:
            box = tuple(box)
            self.trackers.add(self.tracker(), image, box)

        # Определяем, каким объектам соответствуют боксы
        indexes_to_choose = np.apply_along_axis(self._iou, 1, boxes_inter, 
                                   boxes=boxes_detector)
        
        boxes_detector_chosen = boxes_detector[indexes_to_choose, :]
        labels_chosen = labels[indexes_to_choose]
        scores_chosen = scores[indexes_to_choose]

        # Вычисляем признаки объекта
        up_left_corner = boxes_detector_chosen[:, (0, 3)]
        height = boxes_detector_chosen[:, 3] - boxes_detector_chosen[:, 1]
        width = boxes_detector_chosen[:, 2] - boxes_detector_chosen[:, 0]

        # Формируем конечный результат
        objects = []
        for i in range(scores.shape[0]):
            _object = {'up_left_corner': up_left_corner[i, :],
                       'height': height[i],
                       'width': width[i],
                       'score': scores_chosen[i],
                       'label': labels_chosen[i]
            }

            objects.append(_object)

        # Сохраняем данные с текущей итерации
        self.prev_objects = (boxes_detector_chosen, labels_chosen, 
                             scores_chosen)
        
        # Увеличиваем счетчик
        self.iter += 1
        
        return objects

