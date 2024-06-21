import os
import pickle
import cv2
from img2vec_pytorch import Img2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from PIL import Image


img2vec = Img2Vec()
DATA_DIR = 'data'
training_data = []
training_labels = []
validation_data = []
validation_labels = []

for dir_ in os.listdir(DATA_DIR):
    data = []
    labels = []
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_features = img2vec.get_vec(Image.fromarray(img_rgb))
        
        data.append(img_features)
        labels.append(dir_)

    split_index = int(0.8 * len(data))
    training_data.extend(data[:split_index])
    training_labels.extend(labels[:split_index])
    validation_data.extend(data[split_index:])
    validation_labels.extend(labels[split_index:])

training_data = np.array(training_data)
training_labels = np.array(training_labels)
validation_data = np.array(validation_data)
validation_labels = np.array(validation_labels)

model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(training_data, training_labels)

y_pred = model.predict(validation_data)
score = accuracy_score(validation_labels, y_pred)
print(f'Accuracy on validation set: {score}')


with open('model_2.p', 'wb') as f:
    pickle.dump(model, f)

