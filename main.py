import os
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Thiết lập các thông số huấn luyện
batch_size = 32
img_width, img_height = 224, 224
train_data_dir = 'project_folder/train'
test_data_dir = 'project_folder/test'
epochs = 10

# Tạo data generator để tăng cường dữ liệu huấn luyện
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

# Load dữ liệu huấn luyện và kiểm thử
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
)

# Tải mô hình VGG16 đã được huấn luyện sẵn (không bao gồm lớp fully connected)
vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Đóng băng các layer của VGG16 để không cần huấn luyện lại
for layer in vgg16.layers:
    layer.trainable = False

# Xây dựng mô hình mới bằng cách thêm lớp fully connected và dropout
model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Compile và huấn luyện mô hình
model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=len(train_generator), epochs=epochs)

# Lưu mô hình vào file
model.save('cat_dog_classifier.h5')
