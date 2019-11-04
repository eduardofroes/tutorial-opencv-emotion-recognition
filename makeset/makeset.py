import cv2
import glob
import random
from descriptors.FaceLandmarkDescriptor_V2 import get_landmarks

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

def get_files(emotion): #Define function to get file list, randomly shuffle it and split 80/20
    files = glob.glob("/opt/repository/actity/sorted_set/%s/*" %emotion)
    random.shuffle(files)
    training = files[:int(len(files)*0.8)] #get first 80% of file list
    prediction = files[-int(len(files)*0.2):] #get last 20% of file list
    return training, prediction

def make_sets(emotions):
    training_data = []
    training_labels = []
    prediction_data = []
    prediction_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        training, prediction = get_files(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in training:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            data = get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                training_data.append(data['landmarks_vectorised']) #append image array to training data list
                training_labels.append(emotions.index(emotion))
        for item in prediction:
            image = cv2.imread(item)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe_image = clahe.apply(gray)
            data = get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                prediction_data.append(data['landmarks_vectorised'])
                prediction_labels.append(emotions.index(emotion))
    return training_data, training_labels, prediction_data, prediction_labels


def get_files_v2(emotion):
    files = glob.glob("/opt/repository/actity/sorted_set/%s/*" %emotion)
    return files

def make_sets_v2(emotions):
    full_data = []
    full_labels = []
    for emotion in emotions:
        print(" working on %s" %emotion)
        files = get_files_v2(emotion)
        #Append data to training and prediction list, and generate labels 0-7
        for item in files:
            image = cv2.imread(item) #open image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
            clahe_image = clahe.apply(gray)
            data = get_landmarks(clahe_image)
            if data['landmarks_vectorised'] == "error":
                print("no face detected on this one")
            else:
                full_data.append(data['landmarks_vectorised']) #append image array to training data list
                full_labels.append(emotions.index(emotion))
        
    return full_data, full_labels