from config import IMG_SIZE, CLASSES_DIRS, CLASSES_LIST, DATASET_DIR, MAX_SEQUENCE_LENGTH, OVERLAP
import os
import cv2
from process_functions import normalize
from tensorflow.keras.utils import to_categorical

import numpy as np
from tensorflow.keras.applications import VGG16

def get_paths():
    
    videos_paths = []
    labels = []
    
    for class_index, class_name in enumerate(CLASSES_LIST):
        
        for dir in CLASSES_DIRS[class_name]:
            
            print(f"Extracting data from {dir}")
            class_samples = os.path.join(DATASET_DIR, dir)
            
            for sample in os.listdir(class_samples):
                
                if "jpeg" in os.listdir(os.path.join(class_samples, sample)):
                    video_directory = os.path.join(class_samples, sample, "jpeg")
                else:
                    video_directory = os.path.join(class_samples, sample)
                    
                paths = []
                
                for file_name in os.listdir(video_directory):
                    
                    if file_name.lower().endswith(('.jpg')):
                        paths.append(os.path.join(video_directory, file_name))
            
                paths.sort()
                videos_paths.append(paths)
                labels.append(class_index)
                
    return videos_paths, to_categorical(labels)

def sampling_video(frames_list, pred):
    
    video_frames_count = len(frames_list)
    sample_n = int((video_frames_count - OVERLAP)/(MAX_SEQUENCE_LENGTH - OVERLAP))
    n = int(((video_frames_count - OVERLAP)%(MAX_SEQUENCE_LENGTH - OVERLAP))/2)
                
    sample_list = []
    for sample in range(sample_n):
        sample_frames = []
            
        for counter in range(MAX_SEQUENCE_LENGTH):
            
            feature = pred[sample * (MAX_SEQUENCE_LENGTH - OVERLAP) + counter + n]
            feature = feature.reshape(7 * 7 * 512)
            sample_frames.append(feature)   
        
        sample_list.append(np.mean(sample_frames, axis=0))
        
    return sample_list

def process_video(videos_paths, model, resize=(IMG_SIZE, IMG_SIZE)):
    video_list = []
    for i, video in enumerate(videos_paths):
        
        frames_list = np.array([normalize(cv2.imread(path),resize) for path in video])
        pred = model.predict(frames_list)
        print(f"Made prediction for video {i}")
        video_list.append(np.array(sampling_video(frames_list, pred)))
    
    return np.asarray(video_list)
                  
model = VGG16(weights="imagenet", include_top=False)

videos_paths, labels = get_paths()
features = process_video(videos_paths, model)

np.save("labels.npy", labels)
np.save("features.npy", features)

