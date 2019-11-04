# Imports das bibliotecas OpenCV e Dlib.
import cv2
import dlib

video_capture = cv2.VideoCapture(0) # Obtendo acesso a webcam.
detector = dlib.get_frontal_face_detector() # Inicializando o detector de face.
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") # Carregando modelo classificador.
while True:
    ret, frame = video_capture.read() # Capturando frame da camera.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convertendo frame para escala de cinza
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    clahe_image = clahe.apply(gray) # Aplicando conversão da imagem para escala de cinza.
    detections = detector(clahe_image, 1) # Detectando landmarks na imagem.
    for k,d in enumerate(detections): # Iterando sobre as detecções.
        shape = predictor(clahe_image, d) # Obtendo as coordenadas dos landmarks.
        for i in range(1,68): # Listando 68 pontos na face.
            cv2.circle(frame, (shape.part(i).x, shape.part(i).y), 1, (0,0,255), thickness=2) # Desenhando ponto vermelho em cada posição
    cv2.imshow("image", frame) # Mostrando frame com pontos.
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break