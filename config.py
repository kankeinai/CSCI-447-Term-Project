import os
DATASET_DIR = os.path.join("ucf_sports_actions", "ucf action")
CLASSES_DIRS = {
    'Walking' : ['Walk-Front'],
    'Golf' : ['Golf-Swing-Side', 'Golf-Swing-Front', 'Golf-Swing-Back'],
    'Kicking' : ['Kicking-Front', 'Kicking-Side'],
    'Running': ['Run-Side'],
    'Riding-Horse':['Riding-Horse'],
    'Diving':['Diving-Side'],
    'Swing-Side':['Swing-SideAngle'],
    'Swing-Bench':['Swing-Bench'],
    'SkateBoarding':['SkateBoarding-Front'],
    'Lifting':['Lifting']
}
CLASSES_LIST = list(CLASSES_DIRS.keys())
IMG_SIZE = 224

MAX_SEQUENCE_LENGTH = 10
OVERLAP = 5