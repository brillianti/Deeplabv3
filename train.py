# coding=utf-8
import argparse
import glob
import os

import keras
import tensorflow as tf
from keras.callbacks import (CSVLogger, EarlyStopping, ModelCheckpoint,
                             ReduceLROnPlateau)
#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "3" 使用gpu训练

import data
import Models
from Models import build_model
from utils.utils import *

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Deeplabv3")
parser.add_argument(
    "--dataset_name", type=str,
    default="classification_2")  # streetscape(12)(320x640), helen_small(11)(512x512), bbufdataset
parser.add_argument("--n_classes", type=int, default=2)
parser.add_argument("--epochs", type=int, default=60)

parser.add_argument("--input_height", type=int, default=512)
parser.add_argument("--input_width", type=int, default=512)

parser.add_argument('--validate', type=bool, default=True)
parser.add_argument("--resize_op", type=int, default=2)

parser.add_argument("--train_batch_size", type=int, default=2)
parser.add_argument("--val_batch_size", type=int, default=2)

parser.add_argument("--train_save_path", type=str, default="weights/")
parser.add_argument("--resume", type=str, default="")   #ruo wei " " ze yiweiz congtou xunlian
parser.add_argument("--optimizer_name", type=str, default="sgd")
parser.add_argument("--image_init", type=str, default="sub_mean")

args = parser.parse_args()

# 再定义一些keras回调函数需要的参数

# 权重保存
train_save_path = os.path.join(args.train_save_path, args.model_name)
epochs = args.epochs
load_weights = args.resume
mk_if_not_exits(train_save_path)

# patience：没有提升的轮次，即训练过程中最多容忍多少次没有提升
patience = 50
# log_file_path：日志保存的路径
log_file_path = 'weights/%s/log.csv' % args.model_name

# 模型参数
model_name = args.model_name
optimizer_name = args.optimizer_name
image_init = args.image_init

# 数据存储位置
data_root = os.path.join("data", args.dataset_name)
train_images = os.path.join(data_root, "train_image")
train_segs = os.path.join(data_root, "train_label")
train_batch_size = args.train_batch_size
validate = args.validate
if validate:
    val_images = os.path.join(data_root, "test_image")
    val_segs = os.path.join(data_root, "test_label")
    val_batch_size = args.val_batch_size

# 数据参数
n_classes = args.n_classes
input_height = args.input_height
input_width = args.input_width
resize_op = args.resize_op

# modelFns = {
#     'enet': Models.ENet.ENet,
#     'fcn8': Models.FCN8.FCN8,
#     'unet': Models.Unet.Unet,
#     'segnet': Models.Segnet.Segnet,
#     'pspnet': Models.PSPNet.PSPNet,
#     'icnet': Models.ICNet.ICNet,
#     'mobilenet_unet': Models.MobileNetUnet.MobileNetUnet,
#     'mobilenet_fcn8': Models.MobileNetFCN8.MobileNetFCN8,
#     'seunet': Models.SEUNet.SEUnet
# }
# modelFN = modelFns[model_name]
# model = modelFN(n_classes, input_height=input_height, input_width=input_width)

model = build_model(model_name,
                    n_classes,
                    input_height=input_height,
                    input_width=input_width)

# 统计一下训练集/验证集样本数，确定每一个epoch需要训练的iter
images = glob.glob(os.path.join(train_images, "*.jpg")) + \
    glob.glob(os.path.join(train_images, "*.png")) + \
    glob.glob(os.path.join(train_images, "*.jpeg"))

num_train = len(images)

images = glob.glob(os.path.join(val_images, "*.jpg")) + \
    glob.glob(os.path.join(val_images, "*.png")) + \
    glob.glob(os.path.join(val_images, "*.jpeg"))

num_val = len(images)

print(num_train, num_val)

# 模型回调函数
early_stop = EarlyStopping('loss', min_delta=0.1, patience=patience, verbose=1)
reduce_lr = ReduceLROnPlateau('loss',
                              factor=0.01,
                              patience=int(patience / 2),
                              verbose=1)
csv_logger = CSVLogger(log_file_path, append=False)
model_names = os.path.join(train_save_path, '%s.{epoch:02d}-{acc:2f}.hdf5' % (
    args.model_name))
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='loss',
                                   save_best_only=True,
                                   save_weights_only=False)
call_backs = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# compile
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_name,
              metrics=['accuracy'])

if len(load_weights) > 0:
    model.load_weights(load_weights)
print("Model output shape : ", model.output_shape)

output_height = model.outputHeight
output_width = model.outputWidth

# data generator
train_ge = data.imageSegmentationGenerator(train_images, train_segs,
                                           train_batch_size, n_classes,
                                           input_height, input_width, resize_op,
                                           output_height, output_width,
                                           image_init)

if validate:
    val_ge = data.imageSegmentationGenerator(val_images, val_segs,
                                             val_batch_size, n_classes,
                                             input_height, input_width, resize_op,
                                             output_height, output_width,
                                             image_init)

# 开始训练
if not validate:
    model.fit_generator(train_ge,
                        epochs=epochs,
                        callbacks=call_backs,
                        steps_per_epoch=int(num_train / train_batch_size),
                        verbose=1,
                        shuffle=True)
else:
    model.fit_generator(train_ge,
                        validation_data=val_ge,
                        epochs=epochs,
                        callbacks=call_backs,
                        verbose=1,
                        steps_per_epoch=int(num_train / train_batch_size),
                        shuffle=True,
                        validation_steps=int(num_val / val_batch_size))
