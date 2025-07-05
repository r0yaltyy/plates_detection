# Документация проекта YOLOv11 для детекции блюд и приборов на столе

## Датасет и тестовые результаты

Датасет и тестовое видео с результатами обучения на второй версии модели (train1.py с AdamW) доступны по ссылке: https://disk.yandex.ru/d/STPFMJ9gnrsFkw (внутри ZIP-архив с датасетом).
Также, в папке output_ensemble досутпны все видео обработанные ансамблем моделей, а в архиве augmented.zip - аугментированные данные.

## Установка и настройка

### Установка Anaconda

В соответствии с официальной документацией: https://anaconda.com/download

### Установка CUDA и драйверов

В соответствии с официальной документацией: https://developer.nvidia.com/cuda-12-4-0-download-archive.

### Создание и настройка окружения

1. Создайте виртуальное окружение с Python 3.12:

   ```bash
   conda create --name yolo11-env python=3.12 -y
   ```

2. Активируйте окружение:

   ```bash
   conda activate yolo11-env
   ```

3. Установите необходимые библиотеки:

   ```bash
   pip install ultralytics
   pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   pip install albumentations ensemble-boxes numpy
   ```

### Проверка установки

Проверьте доступность CUDA:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

Ожидаемый вывод: имя вашего GPU (например, "NVIDIA GeForce RTX 3090").

## Использование исходных кодов

### frames_parser.py

- **Описание**: Извлекает кадры из всех .mov файлов в указанной директории.
- **Настройка**: Укажите путь к директории с .mov файлами в строке 5.
- **Запуск**: `python frames_parser.py`.

### train.py, train1.py, train2.py

- **Описание**: Три конфигурации обучения модели YOLOv11 (базовая, с AdamW и аугментациями, с yolo11s и повышенным разрешением).
- **Запуск**: `python train.py`, `python train1.py` или `python train2.py` соответственно.

### train3.py, train4.py

- **Описание**: Скрипты для обучения модели с использованием Transfer Learning. `train3.py` использует аугментации (randaugment, mixup, mosaic), а `train4.py` продолжает обучение с заморозкой 10 слоёв на основе весов из директории `/runs/train6`.
- **Запуск**: `python train3.py` или `python train4.py`.

### augment.py, augment_2.py

- **Описание**: Скрипты для аугментации изображений датасета с использованием Albumentations. `augment.py` — изначальная версия с высокими порогами аугментаций, которая привела к сильным шумам и снижению точности. `augment_2.py` — оптимизированная версия с сниженными порогами, использованная для обучения конечных моделей.
- **Запуск**: `python augment.py` или `python augment_2.py`.

### split_augment.py

- **Описание**: Скрипт для распределения аугментированных изображений по директориям `train`, `test` и `valid` в соотношении 80%/10%/10%.
- **Запуск**: `python split_augment.py`.

### cls_count.py

- **Описание**: Простой скрипт для подсчёта количества экземпляров каждого класса в указанной директории разметки (по умолчанию — `valid/labels`).
- **Запуск**: `python cls_count.py`.

### inference.py

- **Описание**: Обычный инференс YOLO.
- **Запуск**: `python inference.py`.

### inference_ensemble.py

- **Описание**: Реализация ансамбля двух моделей (train7 и train8) с использованием Weighted Boxes Fusion (WBF) для обработки видео. Рисует bounding box'ы для каждого класса.
- **Запуск**: `python inference_ensemble.py`.


## Структура датасета

- **/dataset/train/**: 180 изображений для обучения (включая аугментированные).
- **/dataset/test/**: 8 изображений для тестирования.
- **/dataset/valid/**: 10 изображений для валидации.
- **/dataset/train/labels**, **/dataset/test/labels**, **/dataset/valid/labels**: Соответствующие файлы разметки в формате YOLO.
- Общее количество: 198 изображений, аугментированных из 78 разметок (Roboflow с auto-labeling, классы: `['borsh', 'chicken', 'cup', 'cutlery', 'empty cup', 'empty plate', 'meat', 'salad', 'salad balsamic', 'shot', 'soup', 'teapot']`).

## Результаты обучения

В папке `runs` находятся несколько директорий с результатами обучений, включая не самые удачные попытки. Конечные версии моделей:
- `train7`: Оптимизированная модель на основе `augment_2.py`.
- `train8`: Дополнительная оптимизация с Transfer Learning.
Эти модели используются в `inference_ensemble.py` для ансамбля.
```
