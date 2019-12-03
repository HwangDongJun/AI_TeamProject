import math
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from model import CustomModel
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks
from tensorflow.keras import layers
# from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class PlotLearning(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.loss = []
        self.val_loss = []
        self.acc = []
        self.val_acc = []
        self.lr = []
        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))

        print(K.eval(self.model.optimizer.lr))
        self.lr.append(K.eval(self.model.optimizer.lr))
        self.i += 1

        fig = plt.figure(figsize=(10, 10))

        ax1 = fig.add_subplot(1, 3, 1)
        ax1.plot(self.x, self.loss, 'b-', label="train_loss")
        ax1.plot(self.x, self.val_loss, 'r-', label="val_loss")
        ax1.set_title('learning curve')
        ax1.legend(loc=1, prop={'size': 15})

        ax2 = fig.add_subplot(1, 3, 2)
        ax2.plot(self.x, self.acc, 'b-', label="train_acc")
        ax2.plot(self.x, self.val_acc, 'r-', label="val_acc")
        ax2.set_title('accuracy')
        ax2.legend(loc=1, prop={'size': 15})

        ax3 = fig.add_subplot(1, 3, 3)
        ax3.plot(self.x, self.lr, 'b-', label="learning_rate")
        ax3.set_title('learning_rate')
        ax3.legend(loc=1, prop={'size': 15})

        fig.savefig('save/%s.png' % model_name)


def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.5
    epochs_drop = 10
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))

    return max(lrate, 0.000001)


def custom_model(base_model, num_classes):
    x = base_model.output

    custom_model = Model(input=base_model.input, outputs=base)

    return custom_model


if __name__ == "__main__":
    train_graph = tf.Graph()
    train_sess = tf.Session(graph=train_graph)

    keras.backend.set_session(train_sess)
    with train_graph.as_default():
        plot_learning = PlotLearning()

        traindata_dir = "/home/project12/TermP/keras/data/train"
        testdata_dir = "/home/project12/TermP/keras/data/test"

        HEIGHT = 48
        WIDTH = 48

        batch_size = 128
        num_classes = len(os.listdir(traindata_dir))
        epochs = 200

        data_datagen = ImageDataGenerator(
            rescale=1. / 255
        )

        train_generator = data_datagen.flow_from_directory(
            traindata_dir,
            target_size=(HEIGHT, WIDTH),
            batch_size=batch_size,
            class_mode="categorical",
            # subset="training",
            color_mode="grayscale"
        )

        validation_generator = data_datagen.flow_from_directory(
            testdata_dir,
            target_size=(HEIGHT, WIDTH),
            batch_size=batch_size,
            class_mode="categorical",
            # subset="validation",
            color_mode="grayscale"
        )

        '''
        test_generator = test_datagen.flow_from_directory(
            testdata_dir,
            target_size=(HEIGHT, WIDTH),
            batch_size=1,
            class_mode="categorical",
            subset="validation",
            color_mode="grayscale")
        '''

        model_name = 'test1'

        model_init = CustomModel(input_shape=(HEIGHT, WIDTH, 1), num_classes=num_classes, alpha=1.0)
        model = model_init.make_model()

        print(model.summary())

        # SGD_opt = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.1, nesterov=False)
        # Admas_opt = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer='SGD',
                      metrics=['accuracy'])

        filepath = "save/%s.h5" % model_name

        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=0, save_best_only=True, mode='max')

        reduceLR = ReduceLROnPlateau(monitor='val_acc', factor=0.8, patience=5, verbose=1, mode='auto',
                                     min_delta=0.000001, cooldown=0, min_lr=0.000001)

        lrate_decay = LearningRateScheduler(step_decay)

        callbacks_list = [checkpoint, plot_learning, lrate_decay]

        # model.load_weights('./save/veg_sh/mobilnetv1.h5')
        fit_history_mobile = model.fit_generator(
            train_generator,
            steps_per_epoch=train_generator.samples // batch_size,
            validation_data=validation_generator,
            validation_steps=validation_generator.samples // batch_size,
            epochs=epochs,
            callbacks=callbacks_list)

        model.save('test1.h5')

        # saver = tf.train.Saver()
        # saver.save(train_sess, './checkpoints')

        # tf.io.write_graph(train_sess.graph_def, './', 'train.pbtxt')
