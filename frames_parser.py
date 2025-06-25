import os
import cv2

# Укажите путь к директории с .MOV файлами здесь
input_dir = "/home/r0yaltyy/sobes/input"

# Проверяем, существует ли указанная директория
if not os.path.isdir(input_dir):
    print(f"Ошибка: Директория {input_dir} не существует")
    exit(1)

# Создаем папку для сохранения кадров в указанной директории
output_dir = os.path.join(input_dir, "output_frames")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Получаем список всех .MOV файлов в указанной директории
video_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.mov')]

# Интервал извлечения кадров (3 сек)
frame_interval = 3

for video_file in video_files:
    # Формируем полный путь к видео
    video_path = os.path.join(input_dir, video_file)

    # Открываем видео
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Ошибка при открытии файла: {video_file}")
        continue

    # Получаем FPS видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    # Вычисляем интервал кадров для 0.5 сек
    frame_step = int(fps * frame_interval)

    # Счетчик кадров
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Сохраняем кадр, если достигнут нужный интервал
        if frame_count % frame_step == 0:
            # Формируем имя файла: video_name_frame_timestamp.jpg
            timestamp = frame_count / fps
            output_path = os.path.join(
                output_dir,
                f"{os.path.splitext(video_file)[0]}_frame_{timestamp:.1f}s.jpg"
            )
            cv2.imwrite(output_path, frame)
            saved_count += 1

        frame_count += 1

    print(f"Извлечено {saved_count} кадров из {video_file}")
    cap.release()

print("Обработка завершена!")
