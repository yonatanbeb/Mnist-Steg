from __future__ import print_function
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
# the data_sets for learning from encode_mnist.py
from mnist_steg import train_zero, train_parity, train_random
# to document date of current run
from datetime import date


today = date.today()
# adds the date to document the scores
scores = open('learning_scores.txt', 'a+')
scores.write('date of run: ' + today.strftime('%d/%m/%Y') + '\n')

# data_sets from all three labels
data_sets = {
    'zero_label': train_zero,
    'parity_label': train_parity,
    'random_label': train_random
}


def learn_clearance(x_train, y_train, x_test, y_test, type_set):

    batch_size = 128
    num_classes = 10
    epochs = 12
    img_rows, img_cols = 28, 28

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    print('training on ', type_set, ' data_set')
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # transforms labels to categories
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    model = Sequential()

    # adding the layers for the neural network
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # document scores for each data_set
    scores.write('LEARNING SCORES FOR ' + type_set + ' DATA_SET: \n'
                 + '\tTest loss: ' + str(score[0]) + '\n'
                 + '\tTest accuracy: ' + str(score[1]) + '\n')
    return model


# trains all the data_sets
def train():
    data_set_models = {}
    for data_set in data_sets:
        data = data_sets[data_set]
        data_set_models[data['type']] = learn_clearance(data['x_train'], data['y_train'], data['x_test'],
                                                        data['y_test'], data['type'])
    return data_set_models


models = train()

# save model and learnt weights for later use without having to rerun
for model in models:
    models[model].save(model + "_config.h5")
