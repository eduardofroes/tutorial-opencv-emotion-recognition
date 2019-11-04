import cv2
import dlib
import numpy as np
import math

def get_landmarks(image):

    data={}
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    
    detections = detector(image, 1)
    for k,d in enumerate(detections): # Iterando 
        shape = predictor(image, d) # Obtendo as coordenadas dos landmarks.
        xlist = []
        ylist = []
        for i in range(1,68): # Armazenando X e Y em duas listas
            xlist.append(float(shape.part(i).x)) # Armazenando X.
            ylist.append(float(shape.part(i).y)) # Armazenando Y.
        xmean = np.mean(xlist) # Média dos pontos em X (centroide).
        ymean = np.mean(ylist) # Média dos pontos em Y (centroide).
        xcentral = [(x-xmean) for x in xlist] # Calculo da distancia com relação a X.
        ycentral = [(y-ymean) for y in ylist] # Calculo da distancia com relação a Y.
        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w) # Armazenando landmarks
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean,xmean))
            coornp = np.asarray((z,w))
            dist = np.linalg.norm(coornp-meannp) # Calculando distancia entre pontos e centroide.
            landmarks_vectorised.append(dist)  # Armazenando distancia.
            landmarks_vectorised.append((math.atan2(y, x)*360)/(2*math.pi)) # Calculando e armazenando ângulo de incidência.
        data['landmarks_vectorised'] = landmarks_vectorised
        data['centroid'] = [xmean, ymean]
    if len(detections) < 1:
        data['landmarks_vestorised'] = "error"

    return data