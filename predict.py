from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load mô hình đã huấn luyện
img_width, img_height = 224, 224
model = load_model('cat_dog_classifier.h5')

# Đọc ảnh test
img_path = 'project_folder/test/dog/dog_test_image_3.jpg'
img = image.load_img(img_path, target_size=(img_width, img_height))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)

# Dự đoán
result = model.predict(img)
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

print("Predicted:", prediction)
