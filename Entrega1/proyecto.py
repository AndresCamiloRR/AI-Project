import cv2
import os
import mediapipe as mp
import pandas as pd
import math

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Definir el path a la carpeta donde están los videos
video_path = './videos/'
# Lista de nombres de videos y sus respectivas etiquetas (acciones)
video_info = [
    ('sentarse1.mp4', 'sentarse'),
    ('pararse1.mp4', 'pararse'),
    ('caminar1.mp4', 'caminar'),
    ('rotar1.mp4', 'rotar'),
    ('caminar_hacia_atras1.mp4', 'caminar_hacia_atras'),
    ('rotar2.mp4', 'rotar'),
    ('sentarse2.mp4', 'sentarse'),
    ('sentarse3.mp4', 'sentarse'),
    ('pararse2.mp4', 'pararse'),
    ('caminar2.mp4', 'caminar'),
    ('rotar3.mp4', 'rotar'),
    ('caminar_hacia_atras2.mp4', 'caminar_hacia_atras'),
    ('rotar4.mp4', 'rotar'),
    ('sentarse4.mp4', 'sentarse'),
    ('sentarse5.mp4', 'sentarse'),
    ('pararse3.mp4', 'sentarse'),
    ('sentarse6.mp4', 'sentarse'),
    ('pararse4.mp4', 'sentarse'),
    ('sentarse7.mp4', 'sentarse'),
    # Añade más tuplas de (nombre_video, etiqueta) según sea necesario
]

# Ruta para guardar las imágenes de los frames
images_path = './imagenes/'
# Ruta para guardar los datos en CSV y Excel
csv_save_path = './csv_excel/datos.csv'
excel_save_path = './csv_excel/datos.xlsx'

# Lista para almacenar los datos de los landmarks
all_data = []

for video_name, action_label in video_info:
    # Abrir el video
    cap = cv2.VideoCapture(os.path.join(video_path, video_name))
    
    # Obtener la tasa de frames por segundo (FPS)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'FPS del video {video_name}: {fps}')
    
    # Contador de fotogramas
    frame_count = 0
    
    # Crear la carpeta para guardar las imágenes si no existe
    video_folder_name = os.path.splitext(video_name)[0]
    video_images_path = os.path.join(images_path, video_folder_name)
    os.makedirs(video_images_path, exist_ok=True)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Guardar cada frame como imagen
        frame_image_name = os.path.join(video_images_path, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_image_name, frame)
        
        # Convertir la imagen a RGB para MediaPipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        
        # Acceder a los puntos clave
        if results.pose_landmarks:
            # Obtener las coordenadas de los landmarks
            frame_data = {
                'frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                'label': action_label  # Columna de label
            }
            
            for i, landmark in enumerate(results.pose_landmarks.landmark):
                # Obtener coordenadas x, y, z
                x = landmark.x
                y = landmark.y
                z = landmark.z
                # Almacenar coordenadas en un diccionario
                frame_data[f'landmark_{i}_x'] = x
                frame_data[f'landmark_{i}_y'] = y
                frame_data[f'landmark_{i}_z'] = z
            
            # Añadir el diccionario a la lista
            all_data.append(frame_data)

        # Mostrar el frame procesado (opcional)
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1

    cap.release()

# Convertir los datos a un DataFrame de Pandas
df = pd.DataFrame(all_data)

# Guardar los datos en un archivo CSV o Excel
df.to_csv(csv_save_path, index=False)
df.to_excel(excel_save_path, index=False)

print(f"Datos guardados en {csv_save_path} y {excel_save_path}")

cv2.destroyAllWindows()
