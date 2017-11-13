import numpy as np
from scipy.misc import imresize
from PIL import Image

class Batcher:
    def __init__(self):
        self.data_sz = 17844

    def prepare_data(self, i):
        '''
        General function for preparing and processing data
        '''
        im = Image.open('data/{}.png'.format(i)).convert('RGB')
        arr = np.array(im)
        arr = imresize(arr, (72, 72, 3))
        return arr

    def get_data(self, batch_sz=16):
        '''
        General batcher that returns (X, y, meta) * batch_sz
        '''
        counter = 0
        xx = []
        for i in range(0, self.data_sz):
            xx.append(self.prepare_data(i))
            counter += 1
            if counter % batch_sz == 0:
                yield xx
                xx = []
        if xx != []: yield xx