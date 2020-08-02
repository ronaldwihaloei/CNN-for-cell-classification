# -*- coding: utf-8 -*-

import argparse
import matplotlib.pyplot as plt
from scipy import misc
from keras.optimizers import Adam
from keras.utils import to_categorical
import cv2,os
import numpy as np
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dropout
from keras.applications.vgg16 import VGG16
from keras.layers.core import Dense, Flatten
from keras.models import *
from keras.layers import Conv2D,BatchNormalization,Activation,MaxPooling2D,GlobalAveragePooling2D
from keras import regularizers

## Params
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='VGG16', type=str, help='choose a type of model')
parser.add_argument('--finetune', default=True , type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--train_data', default='./data/ce/train/', type=str, help='path of train data')
parser.add_argument('--test_dir', default='./data/ce/test/', type=str, help='directory of test dataset')
parser.add_argument('--epoch', default=10, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--save_every', default=5, type=int, help='save model at every x epoches')
parser.add_argument('--norm_size', default=64, type=str, help='path of pre-trained model')
parser.add_argument('--only_test', default=False, type=bool, help='train and test or only test')
args = parser.parse_args()

num_classes = 3
weight_decay = 0.0005
#x_shape = [32,32,3]

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def Vgg16():
                # Build the network of vgg for 10 classes with massive dropout and weight decay as described in the paper.

        model = Sequential()
        weight_decay = 0.0005
        model.add(Conv2D(64, (3, 3), padding='same',input_shape=[64,64,3],kernel_regularizer=regularizers.l2(weight_decay)))

        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))


        model.add(Conv2D(64, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(256, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(512, (3, 3), padding='same',kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(512,kernel_regularizer=regularizers.l2(weight_decay)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())

        model.add(Dropout(0.5))
        model.add(Dense(3))
        model.add(Activation('softmax'))
        return model

def load_data(filename):
    class_train = []
    label_train = []
    data = []
    for train_class in os.listdir(filename):
        for pic in os.listdir(filename + train_class):
            class_train.append(filename + train_class + '/' + pic)
            label_train.append(train_class)
    temp = np.array([class_train, label_train])
    temp = temp.transpose()
    # shuffle the samples
    np.random.shuffle(temp)
    # after transpose, images is in dimension 0 and label in dimension 1
    image_list = list(temp[:, 0])
    for file in image_list:
        im = misc.imread(file)
        image = cv2.resize(im, (args.norm_size, args.norm_size))
        image = img_to_array(image)
        data.append(image)
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]
    data = np.array(data, dtype="float") / 255.0

    label_list = np.array(label_list)
    label_list = to_categorical(label_list, num_classes=3)

    return data, label_list


def train(aug, trainX, trainY, testX, testY):
    # initialize the model
    print("[INFO] compiling model...")
    #VGG = model.VGG16
    if args.finetune :
        base_model = VGG16(weights= 'imagenet', include_top = False)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation = 'relu')(x)
        predictions = Dense(3, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
    else:
        model = Vgg16()
    opt = Adam(lr=args.lr, decay=args.lr / args.epoch)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")
    H = model.fit_generator(aug.flow(trainX, trainY, batch_size=args.batch_size),
                            validation_data=(testX, testY), steps_per_epoch=len(trainX) // args.batch_size,
                            epochs=args.epoch, verbose=1)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save('./models/2.h5')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = args.epoch
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on cancer cell classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("plot1.png")

if __name__=='__main__':
    train_file_path = args.train_data
    test_file_path = args.test_dir
    trainX , trainY = load_data(train_file_path)
    testX , testY = load_data(test_file_path)
    # construct the image generator for data augmentation
    aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
        height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
        horizontal_flip=True, fill_mode="nearest")
    train(aug,trainX,trainY,testX,testY)