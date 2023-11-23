# Import các thư viện cần thiết
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Đọc dữ liệu kiểm thử và đánh giá mô hình
img_width, img_height = 224, 224
test_data_dir = 'project_folder/test'
batch_size = 32

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Load mô hình đã huấn luyện từ file
model = load_model('cat_dog_classifier.h5')

# Đánh giá mô hình trên tập kiểm thử
score = model.evaluate(test_generator, steps=len(test_generator))
print("Loss:", score[0])
print("Accuracy:", score[1])
