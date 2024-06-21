import pickle
import cv2
import numpy as np
from img2vec_pytorch import Img2Vec
from PIL import Image


model_path = 'model_2.p'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = 'test\hvang2.JPG'

labels_dict = {0: 'trang', 1: 'vang', 2: 'chet'}

img = cv2.imread(image_path)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


img_float32 = img_rgb.astype(np.float32)

img_pil = Image.fromarray(np.uint8(img_float32))

img_features = img2vec.get_vec(img_pil)

img_np = np.array([img_features])

predicted_label = model.predict(img_np)
predicted_character = labels_dict[int(predicted_label[0])]

print(f'Predicted label: {predicted_label[0]}')
print(f'Predicted character: {predicted_character}')

cv2.imshow('Test Image', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
