#bulid DeepstyleGAN model using keras

import  os
import  tensorflow as tf
import  numpy as np
import  keras
from    keras.models import Sequential
from    keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from    keras.optimizers import Adam
from    keras.preprocessing.image import ImageDataGenerator
from    keras.callbacks import ModelCheckpoint, TensorBoard
from    keras.models import load_model
from    keras.utils import plot_model


# define the deepstyleGAN model
def deepstyleGAN(input_shape, n_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    return model



def train(model, train_data_dir, epochs, batch_size, sample_interval, save_dir):
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0002, beta_1=0.5),
                  metrics=['accuracy'])

    
    model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // batch_size,
        callbacks=[tensorboard, checkpoint]
    )

  
    model.save(os.path.join(save_dir, 'model.h5'))


def train_generator(batch_size, train_data_dir):
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    return train_generator

def train_samples(train_data_dir):
    train_samples = len(os.listdir(train_data_dir))
    return train_samples

def validation_generator(batch_size, validation_data_dir):
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    return validation_generator


def validation_samples(validation_data_dir):
    validation_samples = len(os.listdir(validation_data_dir))
    return validation_samples

def tensorboard(log_dir):
    return TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

def checkpoint(save_dir):
    return ModelCheckpoint(os.path.join(save_dir, 'model.h5'), monitor='val_acc', verbose=1, save_best_only=True, mode='max')

def img_height(img_height):
    return img_height

def img_width(img_width):
    return img_width







   

