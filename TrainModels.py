import cv2
import glob
import random
import math
import threading
import numpy as np
import dlib
import itertools
import time
import pandas as pd

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

from makeset.makeset import make_sets_v2

import csv

emotions = ["neutral", "anger", "disgust", "happy", "surprise"]
# emotions = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]

classifiers_linear = []
classifiers_rbf = []
classifiers_poly = []
classifiers_random_forest = []

def trainSVMLinear(data, labels, norm):
    start_time = time.time()
    scores_linear = []
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    folder_count=0
    fileName = "result_linear_%s" % time.time()
    for train_index, test_index in cv.split(data):
      scaler = StandardScaler()
      
      clf_linear = SVC(kernel="linear", probability=True, tol=1e-3)
      
      X_input_train = np.array([data[i] for i in train_index])
      X_input_test = np.array([data[i] for i in test_index])

      if(norm):
        scaler.fit(X_input_train)
        X_input_train = scaler.transform(X_input_train)

        scaler.fit(X_input_test)
        X_input_test = scaler.transform(X_input_test)

      Y_output_train = [labels[i] for i in train_index]
      Y_output_test = [labels[i] for i in test_index]
      
      clf_linear.fit(X_input_train, Y_output_train)
      
      folder_count = folder_count+1
      accuracy_linear = clf_linear.score(X_input_test, Y_output_test)*100  
      save_results_txt(fileName, "Folder %i - Accuracy SVM Linear:%.2f" % (folder_count, accuracy_linear))
      scores_linear.append(accuracy_linear)

      y_pred = clf_linear.predict(X_input_test)
      save_results_txt(fileName, "%s" % confusion_matrix(Y_output_test, y_pred))

      classifiers_linear.append(clf_linear)

    save_results_txt(fileName, "SVM Linear Accurancy:%.2f"  % np.mean(scores_linear))
    save_results_txt(fileName, "Time to train SVM Linear:%s s" % (time.time() - start_time))  

def trainSVMRBF(data, labels, norm):
    start_time = time.time()
    scores_rbf = []
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    folder_count=0
    fileName = "result_rbf_%s" % time.time()
    for train_index, test_index in cv.split(data):
      scaler = StandardScaler()
      
      clf_rbf = SVC(kernel="rbf", probability=True, tol=1e-3, gamma='auto')
      
      X_input_train = np.array([data[i] for i in train_index])
      X_input_test = np.array([data[i] for i in test_index])

      if(norm):
        scaler.fit(X_input_train)
        X_input_train = scaler.transform(X_input_train)

        scaler.fit(X_input_test)
        X_input_test = scaler.transform(X_input_test)

      Y_output_train = [labels[i] for i in train_index]
      Y_output_test = [labels[i] for i in test_index]
      
      clf_rbf.fit(X_input_train, Y_output_train)
      
      folder_count = folder_count+1  
      accuracy_rbf = clf_rbf.score(X_input_test, Y_output_test)*100  
      save_results_txt(fileName, "Folder %i - Accuracy SVM RBF:%.2f" % (folder_count, accuracy_rbf))
      scores_rbf.append(accuracy_rbf)

      y_pred = clf_rbf.predict(X_input_test)
      save_results_txt(fileName, "%s" % confusion_matrix(Y_output_test, y_pred))

      classifiers_rbf.append(clf_rbf)
      
    save_results_txt(fileName, "SVM RBF Accurancy:%.2f"  % np.mean(scores_rbf))
    save_results_txt(fileName, "Time to train SVM RBF:%s s" % (time.time() - start_time))

def trainSVMPoly(data, labels, norm):
    start_time = time.time()
    scores_poly = []
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    folder_count=0
    fileName = "result_poly_%s" % time.time()
    for train_index, test_index in cv.split(data):
      scaler = StandardScaler()

      clf_poly = SVC(kernel='poly', probability=True, tol=1e-3, gamma='auto')
      
      X_input_train = np.array([data[i] for i in train_index])
      X_input_test = np.array([data[i] for i in test_index])

      if(norm):
        scaler.fit(X_input_train)
        X_input_train = scaler.transform(X_input_train)

        scaler.fit(X_input_test)
        X_input_test = scaler.transform(X_input_test)

      Y_output_train = [labels[i] for i in train_index]
      Y_output_test = [labels[i] for i in test_index]
      
      clf_poly.fit(X_input_train, Y_output_train)
      
      folder_count = folder_count+1  
      accuracy_poly = clf_poly.score(X_input_test, Y_output_test)*100  
      save_results_txt(fileName, "Folder %i - Accuracy SVM Poly:%.2f" % (folder_count, accuracy_poly))
      scores_poly.append(accuracy_poly)

      y_pred = clf_poly.predict(X_input_test)
      save_results_txt(fileName, "%s" % confusion_matrix(Y_output_test, y_pred))

      classifiers_poly.append(clf_poly)
      
    save_results_txt(fileName, "SVM poly Accurancy:%.2f"  % np.mean(scores_poly))
    save_results_txt(fileName, "Time to train SVM Poly:%s s" % (time.time() - start_time))    

def trainRandomForest(data, labels, norm):
    start_time = time.time()
    scores_random_forest = []
    cv = KFold(n_splits=5, random_state=42, shuffle=True)
    folder_count=0
    fileName = "result_forest_%s" % time.time()

    for train_index, test_index in cv.split(data):
      scaler = StandardScaler()
      
      clf_random_forest = RandomForestRegressor(
           bootstrap=True, max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_weight_fraction_leaf=0.0,
           n_estimators=100, n_jobs=1, oob_score=False,
           verbose=0, warm_start=False
      )
      
      X_input_train = np.array([data[i] for i in train_index])
      X_input_test = np.array([data[i] for i in test_index])

      if(norm):
        scaler.fit(X_input_train)
        X_input_train = scaler.transform(X_input_train)

        scaler.fit(X_input_test)
        X_input_test = scaler.transform(X_input_test)     

      Y_output_train = [labels[i] for i in train_index]
      Y_output_test = [labels[i] for i in test_index]
      
      clf_random_forest.fit(X_input_train, Y_output_train)
      
      folder_count = folder_count+1
      accuracy_random_forest = clf_random_forest.score(X_input_test, np.array(Y_output_test).reshape(len(Y_output_test), 1))*100  

      save_results_txt(fileName, "Folder %i - Accuracy Random Forest:%.2f" % (folder_count, accuracy_random_forest))
      scores_random_forest.append(accuracy_random_forest)

      y_pred = [round(y) for y in clf_random_forest.predict(X_input_test)]
      save_results_txt(fileName, "%s" % confusion_matrix(Y_output_test, np.array(y_pred).reshape(len(y_pred), 1)))

      classifiers_random_forest.append(clf_random_forest)

    save_results_txt(fileName, "Random Forest Accurancy:%.2f"  % np.mean(scores_random_forest))
    save_results_txt(fileName, "Time to train Random Forest:%s s" % (time.time() - start_time))  

def save_features_csv(data, labels):
    with open('data_%s.csv' %(time.time()), mode='w') as extracted_file:
        extracted_writer = csv.writer(extracted_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        indexes = []

        for i in np.arange(268):
          indexes.append(str(i))

        indexes.append('label')
        extracted_writer.writerow(indexes)
      
        for i in range(len(data)):
          full_data = data[i]
          full_data.append(labels[i])
          extracted_writer.writerow(full_data)

def save_results_txt(fileName, row):
   with open(fileName, mode='a+') as result:
      result.write(row + '\n')
   result.close()

if __name__ == "__main__":  
    fileName = "extracted_features_data_5_emotions.csv"
    data=[]
    labels=[]    

    if(fileName == None):
      data, labels = make_sets_v2(emotions)
      save_features_csv(data, labels)
    else:
      data_frame = pd.read_csv('./%s' % fileName, delimiter=",", encoding='utf-8', error_bad_lines=False)
      all_data = data_frame.to_numpy()
      data =  all_data[:, :-1]
      labels = all_data[:, -1]

    svmLinearThread = threading.Thread(target=trainSVMLinear, args=(data, labels,True,))
    svmRBFThread = threading.Thread(target=trainSVMRBF, args=(data, labels,True,))
    svmPolyThread = threading.Thread(target=trainSVMPoly, args=(data, labels,True,))
    randomForestThread = threading.Thread(target=trainRandomForest, args=(data, labels,True,))

    randomForestThread.start()
    svmPolyThread.start()
    svmLinearThread.start()
    svmRBFThread.start()
    