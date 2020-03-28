from keras.models import load_model
from keras import backend as K
from PIL import Image
import numpy as np
from mnist_steg import test_zero, test_parity, test_random


# to keep score of testing
class Accuracy:
    def __init__(self):
        self.correct = 0
        self.incorrect = 0

    def prediction(self, correct):
        if correct:
            self.correct += 1
        else:
            self.incorrect += 1


test_sets = {
    'zero_label': test_zero,
    'parity_label': test_parity,
    'random_label': test_random
}
img_rows, img_cols = 28, 28
x_try = {}


for set in test_sets:
    x_data = test_sets[set]['x']
    if K.image_data_format() == 'channels_first':
        x_try[set] = x_data.reshape(x_data.shape[0], 1, img_rows, img_cols)
    else:
        x_try[set] = x_data.reshape(x_data.shape[0], img_rows, img_cols, 1)
    x_try[set] = x_try[set].astype('float32')
    x_try[set] /= 255


run = True
while run:
    data = input('select one of the following clearance types: {zero_label, parity_label, random_label} \n '
                 'enter END to exit. \n')
    if data == 'END':
        break
    model = load_model('saved_models//' + data)    # path to data model
    x = x_try[data]

    num = int(input('choose num: '))
    image = test_sets[data]['x'][num]
    clearance = image[0][0]

    display = Image.fromarray(image)
    display.show()

    x_for_prediction = np.expand_dims(x[num], 0)
    prediction = model.predict(x_for_prediction)
    # change y so that it shows the regular image
    print('digit in image: ', test_sets[data]['y'][num], '\n'
          'clearance level: ', clearance, '\n'
          'prediction: ', np.argmax(prediction[0]), '\n')
