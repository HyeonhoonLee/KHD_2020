#main23-7-5: 가로 1/5~4/5 세로 1/4~3/4, Densenet+Hands, cutout(5), val_ratio 0.1, batch 32, epoch 300
#main23-7-6: 가로 1/5~4/5 세로 1/4~3/4,datasetoversampling 2-3-4
#main23-7-7: 가로 1/5~4/5 세로 1/4~3/4,datasetoversampling 3-4-5
#main23-7-8: 가로 1/5~4/5 세로 1/4~3/4, oversampling 2-3-4 label_smoothing(classtolabel 수정, 0.1)
#main23-7-9: oversampling 2-3-4, cutout 10,  
#main23-7-9를 VGG19로만 바꿈... 원래대로 하려면 회귀 필요.

## Next
#main2: label_smoothing  
#main2: 여러 모델 해보고 최종적으로 앙상블.https://hwiyong.tistory.com/100

import os, sys
import argparse
import time
import random
import cv2
import numpy as np
import keras
# from keras_applications.vgg19 import VGG19
# from keras.applications.resnet50 import ResNet50
# from keras.applications import InceptionResNetV2
# from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121
from keras.utils import np_utils
#from sklearn.model_selection import train_test_split
from keras import layers, optimizers

from keras.models import Model
from keras.models import load_model
from keras.layers import (
    Input, Activation, Dense, Flatten, Conv2D, MaxPooling2D,
    GlobalAveragePooling2D, AveragePooling2D, BatchNormalization, add)
import keras.regularizers as regulizers
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping ###EarlyStopping 추가
from keras.utils.training_utils import multi_gpu_model
import keras.backend.tensorflow_backend as K
### Data Augmentation을 위해 추가.
from keras.preprocessing.image import ImageDataGenerator

import nsml
from nsml.constants import DATASET_PATH, GPU_NUM


IMSIZE = 360, 180

# crop_h, crop_w = 120, 0
VAL_RATIO = 0.1
RANDOM_SEED = 844
weights = np.array([1,6,13,26])
smoothing=0.1

def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

import tensorflow as tf
def focal_loss(y_true, y_pred):
    gamma = 2.0
    alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def cutout(images, cut_length):
    """
    Perform cutout augmentation from images.
    :param images: np.ndarray, shape: (N, H, W, C).
    :param cut_length: int, the length of cut(box).
    :return: np.ndarray, shape: (N, h, w, C).
    """

    H, W, C = images.shape[1:4]
    augmented_images = []
    for image in images:    # image.shape: (H, W, C)
        image_mean = int(image.mean(keepdims=True))
        image -= image_mean

        mask = np.ones((H, W, C), np.float32)

        y = np.random.randint(H)
        x = np.random.randint(W)
        length = cut_length

        y1 = np.clip(y - (length // 2), 0, H)
        y2 = np.clip(y + (length // 2), 0, H)
        x1 = np.clip(x - (length // 2), 0, W)
        x2 = np.clip(x + (length // 2), 0, W)

        mask[y1: y2, x1: x2] = 0.
        image = image * mask

        image += image_mean
        augmented_images.append(image)

    return np.stack(augmented_images)    # shape: (N, h, w, C)

def bind_model(model):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        print('model saved!')

    def load(dir_name):
        model.load_weights(os.path.join(dir_name, 'model'))
        print('model loaded!')

    def infer(data):            # test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        X = ImagePreprocessing(data)
        X = np.array(X)
        X = np.expand_dims(X, axis=-1)
        ###pred = model.predict_classes(X)  # 모델 예측 결과: 0-3
        ### To use functional API
        y_prob = model.predict(X)
        y_classes = y_prob.argmax(axis=-1)
        pred = y_classes
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)


def Class2Label(cls):
    lb = [0] * 4
    lb[int(cls)] = 1
    return lb

def DataLoad(imdir):
    impath = [os.path.join(dirpath, f) for dirpath, dirnames, files in os.walk(imdir) for f in files if all(s in f for s in ['.jpg'])]
    img = []
    lb = []
    print('Loading', len(impath), 'images ...')
    for i, p in enumerate(impath):
        img_whole = cv2.imread(p, 0)
        h, w = img_whole.shape
        h, w_ = h, w//2
        l_img = img_whole[:, w_:2*w_]
        r_img = img_whole[:, :w_]
        # l_img = img_whole[:, int(0.5*w_):w_]
        # r_img = img_whole[:, w_:int(1.5*w_)]
        _, l_cls, r_cls = os.path.basename(p).split('.')[0].split('_')
        if l_cls=='0' or l_cls=='1' or l_cls=='2' or l_cls=='3':
            img.append(l_img);      lb.append(Class2Label(l_cls))
        if r_cls=='0' or r_cls=='1' or r_cls=='2' or r_cls=='3':
            img.append(r_img);      lb.append(Class2Label(r_cls))
    print(len(img), 'data with label 0-3 loaded!')
    return img, lb


def ImagePreprocessing(img):
    # 자유롭게 작성
    h, w = IMSIZE
    print('Preprocessing ...')
    for i, im, in enumerate(img):
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        tmp_pre = clahe.apply(im)
        tmp = cv2.resize(tmp_pre, dsize=(w, h), interpolation=cv2.INTER_AREA)
        tmp = tmp / 255.
        ###image slice vertical
        # h_,w_ = h//2, w//2
        # tmp_l = tmp[int(0.5*h_):int(1.5*h_), :w_]
        ###image slice
        h_,w = h//2, w
        tmp_c = tmp[int(0.5*h_):int(1.5*h_), int(0.2*w):int(0.8*w)]
        img[i] = tmp_c / 255.
    print(len(img), 'images processed!')
    
    return img


# Function Definitions
def ParserArguments(args):
    # Setting Hyperparameters
    args.add_argument('--epoch', type=int, default=300)          # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=16)      # batch size 설정
    args.add_argument('--learning_rate', type=float, default=1e-4)  # learning rate 설정
    args.add_argument('--num_classes', type=int, default=4)     # 분류될 클래스 수는 4개

    # DO NOT CHANGE (for nsml)
    args.add_argument('--mode', type=str, default='train', help='submit일 때 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()
    return config.epoch, config.batch_size, config.num_classes, config.learning_rate, config.pause, config.mode

if __name__ == '__main__':
    args = argparse.ArgumentParser()

    nb_epoch, batch_size, num_classes, learning_rate, ifpause, ifmode = ParserArguments(args)

    seed = 844
    np.random.seed(seed)
    """ Model """
    h, w = IMSIZE

    # model = load_model('./brucechou1983_CheXNet_Keras_0.3.0_weights.h5')
    
    # customized model
    orig_model = DenseNet121(include_top=False, weights='imagenet', pooling='avg')

    dense_input = Input(shape=(h//2, int(0.6*w), 1))
    dense_filter = Conv2D(3, 3, padding='valid')(dense_input)
    x = orig_model(dense_filter)
    output = Dense(num_classes, activation='softmax')(x)
    new_model=Model(dense_input, output)

    frozen_layers, trainable_layers = [], []
    for layer in new_model.layers:
        if isinstance(layer, keras.layers.Conv2D):
            layer.trainable = False
            frozen_layers.append(layer.name)
        else:
            if len(layer.trainable_weights) > 0:
                # We list as "trainable" only the layers with trainable parameters.
                trainable_layers.append(layer.name)

    predictions = new_model.output

    dense121_freeze = Model(dense_input, predictions)
    model = dense121_freeze

    # this is the model we will train    
    # sgd = optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    adam = optimizers.Adam(lr=3e-4)
    # adam = optimizers.Adam(lr=1e-4)
    # nadam = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
    
    ### loss definition
    loss = weighted_categorical_crossentropy(weights)
    model.compile(optimizer=adam, loss=loss, metrics=['categorical_accuracy', get_f1])

    bind_model(model)

    if ifpause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if ifmode == 'train':  ### training mode일 때
        print('Training Start...')
        images, labels = DataLoad(os.path.join(DATASET_PATH, 'train'))
        images = ImagePreprocessing(images)

        ## data 섞기
        images = np.array(images)
        images = np.expand_dims(images, axis=-1)
        labels = np.array(labels)
        
        dataset0 = [[X, Y] for X, Y in zip(images, labels) if Y[0]==1]
        dataset1 = [[X, Y] for X, Y in zip(images, labels) if Y[1]==1]
        dataset2 = [[X, Y] for X, Y in zip(images, labels) if Y[2]==1]
        dataset3 = [[X, Y] for X, Y in zip(images, labels) if Y[3]==1]
        
        ## data를 trainin과 validation dataset으로 나누기 
        tmp0 = int(len(dataset0) * VAL_RATIO)  #Y->labels
        tmp1 = int(len(dataset1) * VAL_RATIO)
        tmp2 = int(len(dataset2) * VAL_RATIO)
        tmp3 = int(len(dataset3) * VAL_RATIO)
        
        dataset0_train = dataset0[tmp0:]
        dataset0_val = dataset0[:tmp0]
        
        dataset1_train = dataset1[tmp1:]
        dataset1_val = dataset1[:tmp1]
        
        dataset2_train = dataset2[tmp2:]
        dataset2_val = dataset2[:tmp2]
        
        dataset3_train = dataset3[tmp3:]
        dataset3_val = dataset3[:tmp3]
        
        for i in range(2):
            x1 = dataset1_train.copy()
            dataset1_train = np.concatenate((dataset1_train,x1),axis=0)
        for i in range(3):
            x2 = dataset2_train.copy()
            dataset2_train = np.concatenate((dataset2_train,x2),axis=0)
        for i in range(4):
            x3 = dataset3_train.copy()
            dataset3_train = np.concatenate((dataset3_train,x3),axis=0)
        
        
        dataset_train = np.concatenate((dataset0_train, dataset1_train, dataset2_train, dataset3_train), axis=0)
        dataset_val = np.concatenate((dataset0_val, dataset1_val, dataset2_val, dataset3_val), axis=0)
        
        random.shuffle(dataset_train)
        random.shuffle(dataset_val)
        X_train = np.array([n[0] for n in dataset_train])
        Y_train = np.array([n[1] for n in dataset_train])
        X_val = np.array([n[0] for n in dataset_val])
        Y_val = np.array([n[1] for n in dataset_val])

        ### using cutout
        X_train_cut = X_train.copy()
        cutout(X_train_cut, 10)
        X_train = np.concatenate((X_train, X_train_cut),axis=0)
        # concatenate: labels
        Y_train_cut= Y_train.copy()
        Y_train = np.concatenate((Y_train, Y_train_cut),axis=0)
   
        
        #Data Augmentatioin
        
        data_gen_args = dict(featurewise_center=False,
                     featurewise_std_normalization=False,
                     rotation_range=10,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=False,
                     fill_mode='nearest')
        train_datagen = ImageDataGenerator(**data_gen_args)
        val_datagen = ImageDataGenerator(featurewise_center=False)
        # cutout_datagen = ImageDataGenerator(**data_gen_args)
      
        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = seed
        train_datagen.fit(X_train, augment=True, seed=seed)
        val_datagen.fit(X_val, augment=False, seed=seed)
        # cutout_datagen.fit(X_train_cut, augment=False, seed=seed)
       
        train_generator = train_datagen.flow(x=X_train, y=Y_train, shuffle=True, batch_size=batch_size, seed=seed)
        validation_generator = val_datagen.flow(x=X_val, y=Y_val, shuffle=True, batch_size=batch_size, seed=seed)
        # cutout_generator = cutout_datagen.flow(x=X_train_cut, y=Y_train_cut, shuffle=True, batch_size=batch_size, seed=seed)
   
        # combine generators into one which yields image and masks
        # train_generator = zip(image_generator, cutout_generator)
# 
        """ Callback """
        # monitor = 'categorical_accuracy'
        # monitor = 'val_loss'
        reduce_lr = ReduceLROnPlateau(monitor='loss', patience=3)
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        
        """ Training loop """
        #STEP_SIZE_TRAIN = len(images) // batch_size    ##X -> images
        STEP_SIZE_TRAIN = len(X_train) // batch_size
        ## validation_step 추가
        STEP_SIZE_VAL = len(X_val) // batch_size
        print('\n\nSTEP_SIZE_TRAIN = {}\n\n'.format(STEP_SIZE_TRAIN))
        
        ### fitting....
        t0 = time.time()
        for epoch in range(nb_epoch):
            t1 = time.time()
            print("### Model Fitting.. ###")
            print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
            print('check point = {}'.format(epoch))

            # for no augmentation case
            # class_weight = {0: 1.,
            #     1: 6.,
            #     2: 13.,
            #     3: 26.}
            hist = model.fit_generator(train_generator,
                             validation_data=validation_generator,
                             steps_per_epoch=STEP_SIZE_TRAIN,
                             # epochs=epoch,
                             validation_steps=STEP_SIZE_VAL,
                             # batch_size=batch_size,
                             # initial_epoch=epoch,
                             # class_weight=class_weight,
                             callbacks=[reduce_lr, early_stop]
                                      )
                             # shuffle=False

            print(hist.history)
            train_acc = hist.history['categorical_accuracy'][0]
            train_loss = hist.history['loss'][0]
            val_acc = hist.history['val_categorical_accuracy'][0]
            val_loss = hist.history['val_loss'][0]
            f1_score = hist.history['get_f1'][0]
            val_f1_score = hist.history['val_get_f1'][0]
            
            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc, val_loss=val_loss,
                        val_acc=val_acc, f1_score=f1_score, val_f1_score=val_f1_score)
            nsml.save(epoch)
            print('Training time for one epoch : %.1f' % (time.time() - t1))
        print('Total training time : %.1f' % (time.time() - t0))
# cd nsml_client.linux.amd64.hack
# export PATH=$PATH:/workspace/ENTropy/nsml_client.linux.amd64.hack