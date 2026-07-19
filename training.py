import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_path = "/content/dataset/train"
test_path = "/content/dataset/test"

datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.2,
)

train_data = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="training",
)

val_data = datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="binary",
    subset="validation",
)
