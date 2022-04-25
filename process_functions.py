import cv2
import numpy as np
from config import CLASSES_LIST



def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def normalize(frame, resize):
    frame = crop_center_square(frame)
    resized_frame = cv2.resize(frame, resize)
    normalized_frame = resized_frame / 255
    return normalized_frame


def repair_data(X, Y):
    np.random.seed(11)
    x = []
    y = []

    for feature, label in zip(X, Y):
        x.extend(feature)
        y.extend([label]*len(feature))
    x = np.array(x)
    y = np.array(y)
    idx = np.random.permutation(len(y))
    
    return x[idx], y[idx]

def predict_random_forest(model, features, X_test_idx):
    return [np.mean(model.predict(features[video]), axis=0) for video in X_test_idx]

def predict_SVC(model_SVC, features, X_test_idx):
    pred = [model_SVC.predict(features[video]) for video in X_test_idx]
    prediction_SVC = []

    for video in pred:
        prob = np.zeros(len(CLASSES_LIST))
        class_idx, class_num = np.unique(video, return_counts=True)
        for n, count in zip(class_idx, class_num):
            prob[n] = count/len(video)
        prediction_SVC.append(prob)
        
    return prediction_SVC

