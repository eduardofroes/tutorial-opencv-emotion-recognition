# Imports das bibliotecas OpenCV e Dlib.
import cv2
import dlib
from descriptors.FaceLandmarkDescriptor_V2 import get_landmarks

video_capture = cv2.VideoCapture(0) #Obtendo acesso a webcam.
while True:
    ret, frame = video_capture.read() # Capturando frame da camera.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convertendo frame para escala de cinza.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray) # Aplicando conversão para escala de cinza.
    data = get_landmarks(clahe_image) # Capturando landmarks.

    pattern = data['landmarks_vectorised'] #Capturando os padrões gerados.
    centroid = data['centroid'] #Capturando o centroide da face.

    for i in range(int(len(pattern)/4)):
        cv2.circle(frame, (int(pattern[i*4]), int(pattern[i*4 + 1])), 1, (0,0,255), thickness=2) # Desenhando ponto vermelho em cada posição.
        cv2.line(frame,(int(pattern[i*4]),int(pattern[i*4 + 1])),(int(centroid[0]), int(centroid[1])),(0,0,255),1) # Desenhando vetores.
        
    cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 1, (255,0,0), thickness=2) # Desenhando centroide.

    cv2.imshow("image", frame) # Demonstrndo frame com landmarks.
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break