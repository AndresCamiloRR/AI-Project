import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import cv2
import mediapipe as mp
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Cargar el modelo desde el archivo .pkl
model = joblib.load("modelo_xgb.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Inicializar Mediapipe
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Lista para almacenar los landmarks de los frames
landmarks_list = []

# Inicializar el escalador
scaler = MinMaxScaler()

# Función para preprocesar los landmarks
def preprocess_landmarks(landmarks):
    data = []
    for landmark in landmarks.landmark:
        data.extend([landmark.x, landmark.y, landmark.z])
    return data

# Capturar video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # Procesar la imagen con Mediapipe
    results = holistic.process(image)

    # Convertir la imagen de vuelta a BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Dibujar los landmarks en la imagen
    mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

    # Extraer landmarks si están disponibles
    if results.pose_landmarks:
        landmarks = preprocess_landmarks(results.pose_landmarks)
        landmarks_list.append(landmarks)

        # Procesar los landmarks en grupos de 6 frames
        if len(landmarks_list) >= 16:
            aggregated_data = {}

            # Normalizar los landmarks
            recent_landmarks = scaler.fit_transform(landmarks_list[-16:])

            # Calcular las diferencias entre los puntos de los frames
            for j in range(33):  # Hay 33 landmarks en MediaPipe Pose
                aggregated_data[f'landmark_{j}_x_diff'] = recent_landmarks[-1][j*3] - recent_landmarks[0][j*3]
                aggregated_data[f'landmark_{j}_y_diff'] = recent_landmarks[-1][j*3+1] - recent_landmarks[0][j*3+1]
                aggregated_data[f'landmark_{j}_z_diff'] = recent_landmarks[-1][j*3+2] - recent_landmarks[0][j*3+2]

            
            # Calcular las diferencias entre los landmarks 26 y 25
            row_15_1= recent_landmarks[-1][24*3] - recent_landmarks[-1][23*3]
            row_15_2= recent_landmarks[-1][24*3+1] - recent_landmarks[-1][23*3+1]
            row_15_3= recent_landmarks[-1][24*3+2] - recent_landmarks[-1][23*3+2]

            row_0_1= recent_landmarks[0][24*3] - recent_landmarks[0][23*3]
            row_0_2= recent_landmarks[0][24*3+1] - recent_landmarks[0][23*3+1]
            row_0_3 = recent_landmarks[0][24*3+2] - recent_landmarks[0][23*3+2]

            aggregated_data['landmark_24_23_x_diff'] = row_15_1 - row_0_1
            aggregated_data['landmark_24_23_y_diff'] = row_15_2 - row_0_2
            aggregated_data['landmark_24_23_z_diff'] = row_15_3 - row_0_3

            # Convertir el diccionario a un DataFrame
            result_df_2 = pd.DataFrame([aggregated_data])

            landmarks_to_drop = []
            for i in range(1,11):
                landmarks_to_drop.extend([f'landmark_{i}_x_diff', f'landmark_{i}_y_diff', f'landmark_{i}_z_diff'])

            result_df_2.drop(columns=landmarks_to_drop, inplace=True)

            # Hacer una predicción con el modelo
            prediction_numeric = model.predict(result_df_2)
            prediction_proba = model.predict_proba(result_df_2)

            # Convertir la predicción numérica a la etiqueta de texto
            prediction_label = label_encoder.inverse_transform(prediction_numeric)

            
            # Calcular el porcentaje de confianza de la predicción
            confidence_score = np.max(prediction_proba) * 100

            # Mostrar la predicción en la pantalla
            if((confidence_score>60 and prediction_label[0] != 'rotar') or confidence_score>85):
    
                cv2.putText(image, f'Predicción: {prediction_label[0]} ({confidence_score:.2f}%)', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Mostrar la imagen con la predicción
    cv2.imshow('Predicción en tiempo real', image)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()