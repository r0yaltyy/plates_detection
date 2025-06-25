# Документация проекта для детекции блюд на столе  с использованием YOLOv11

## Датасет и тестовые результаты

Датасет и тестовое видео с результатами обучения на второй версии модели (train1.py с AdamW) доступны по ссылке: https://disk.yandex.ru/d/STPFMJ9gnrsFkw (внутри ZIP-архив с датасетом и описанием аугментации).

## Установка и настройка

### Установка Anaconda

В соответствии с официальной документацией:  https://anaconda.com/download

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

### inference.py

- **Описание**: Выполняет инференс на видео по указанному path.
- **Запуск**: `python inference.py`.

## Структура датасета

- **/dataset/train/**: 180 изображения для обучения.
- **/dataset/test/**: 8 изображения для тестирования.
- **/dataset/val/**: 10 изображения для валидации.
- **/dataset/train/labels - разметка**
- **/dataset/test/labels - разметка** 
- **/dataset/val/labels - разметка**
- Общее количество: 198 изображений, аугментированных из 78 разметок (Roboflow с auto-labeling, классы: \['borsh', 'chicken', 'cup', 'cutlery', 'empty cup', 'empty plate', 'meat', 'salad', 'salad balsamic', 'shot', 'soup', 'teapot'\]).

## Результаты обучения

В папке `runs` находятся три директории с результатами обучения, соответствующие файлам `train.py`, `train1.py` и `train2.py`
