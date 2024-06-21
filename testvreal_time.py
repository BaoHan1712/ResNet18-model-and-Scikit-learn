import pickle
import cv2
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image
import time

model_path = 'model_2.p'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()


labels_dict = {0: 'trang', 1: 'vang', 2: 'chet'}

cap = cv2.VideoCapture(0)  
pTime = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    img_float32 = img_rgb.astype(np.float32)

    img_pil = Image.fromarray(np.uint8(img_float32))

    img_features = img2vec.get_vec(img_pil)
    img_np = np.array([img_features])

    predicted_label = model.predict(img_np)
    predicted_character = labels_dict[int(predicted_label[0])]
    
    cTime=time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),3)

    cv2.putText(frame, f'Predicted character: {predicted_character}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Test Image', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
