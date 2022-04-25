import pickle
import numpy as np
from datetime import datetime as dt
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score, accuracy_score

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from process_functions import repair_data, predict_SVC, predict_random_forest

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import KFold

seed_constant = 89
np.random.seed(seed_constant)


def construct_random_forest():
    return RandomForestClassifier(n_estimators=5, random_state=seed_constant, n_jobs = 3)
    
def construct_SVC():
    return OneVsRestClassifier(SVC(kernel="linear", class_weight="balanced"))  
    

def k_fold_validation(k, model_construct, predict, SVC=True):
    
    print(f"K-fold Validation, k={k}")
    kfold = KFold(k, shuffle = True, random_state=11)
    counter = 1
    
    acc_scores = []
    f1_scores = []
    
    for train_index, test_index in kfold.split(X_train_idx):
        
        model = model_construct()
        X_train_val, y_train_val = repair_data(features[train_index], labels[train_index])
        
        if SVC: 
            y_train_val =  np.argmax(y_train_val, axis = 1)
            
        model.fit(X_train_val, y_train_val)
        
        true = np.argmax(labels[test_index], axis=1)
        prediction = np.argmax(predict(model, features, test_index), axis=1)
        
        
        f1 = f1_score(true, prediction, average='weighted')
        acc = accuracy_score(true, prediction)
        
        acc_scores.append(acc)
        f1_scores.append(f1)
        
        print(f"Validation with {counter}-fold, f1-score: {f1}, accuracy: {acc}")
        
        counter+=1
        
    print(f"Average f1-score: {np.mean(f1_scores)}, average accuracy: {np.mean(acc_scores)}")
    
    return np.array([f1_scores, acc_scores])

features = np.load("features_20_frames_size_300.npy", allow_pickle=True)
labels = np.load("labels.npy")

X_train_idx, X_test_idx, y_train, y_test = train_test_split(np.arange(len(features)), labels, stratify=labels, shuffle=True, test_size=0.33,  random_state = 11)
X_train, Y_train = repair_data(features[X_train_idx], y_train)
true = np.argmax(y_test, axis=1)

print("Created OneVsRest SVC Classifier")  
model_SVC = OneVsRestClassifier(SVC(kernel="linear", class_weight="balanced"))    
model_SVC_scores = k_fold_validation(7, construct_SVC, predict_SVC)
model_SVC.fit(X_train, np.argmax(Y_train, axis = 1))
print("Predicting for test data set")
prediction_SVC = np.argmax(predict_SVC(model_SVC, features, X_test_idx), axis=1)
perfomance = f1_score(true, prediction_SVC, average='weighted')

folder_name = f'model_SVC_{perfomance}_{dt.now()}'
os.mkdir(folder_name)
pickle.dump(model_SVC, open(os.path.join(folder_name, f"model_SVC_{perfomance}.sav"), "wb"))
np.save(os.path.join(folder_name, "model_SVC_scores.npy"), model_SVC_scores)

print("Confusion matrix")
print(confusion_matrix(true, prediction_SVC))
print("Classification report")
print(classification_report(true, prediction_SVC, digits=10))

print("Created Random Forest Classifier")  
model_random_forest = RandomForestClassifier(n_estimators=3, random_state=seed_constant, n_jobs = 3)    
model_random_forest_scores = k_fold_validation(7, construct_random_forest, predict_random_forest, False)
model_random_forest.fit(X_train, Y_train)
print("Predicting for test data set")
prediction_random_forest = np.argmax(predict_random_forest(model_random_forest, features, X_test_idx), axis=1)
perfomance = f1_score(true, prediction_random_forest, average='weighted')

pickle.dump(model_random_forest, open(f"model_random_forest_{perfomance}_{dt.now()}.sav", "wb"))
folder_name = f'model_random_forest_{perfomance}_{dt.now()}'
os.mkdir(folder_name)
pickle.dump(model_random_forest, open(os.path.join(folder_name, f"model_random_forest_{perfomance}.sav"), "wb"))
np.save(os.path.join(folder_name, "mode_random_forest_scores.npy"), model_random_forest_scores)

print("Confusion matrix")
print(confusion_matrix(true, prediction_random_forest))
print("Classification report")
print(classification_report(true, prediction_random_forest, digits=10))