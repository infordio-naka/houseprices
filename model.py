import os
import gc
import numpy                      as     np
from   readData                   import ReadData
from   keras.models               import Sequential, load_model
from   keras.layers.core          import Dense, Activation
from   keras.layers.recurrent     import LSTM
from   keras.layers.normalization import BatchNormalization
from   keras.optimizers           import RMSprop, SGD

class Model(object):
    def __init__(self):
        self.model = None

    def create_model(self, input_shape, output_shape):
        self.model = Sequential()
        self.model.add(Dense(1024, input_shape=(input_shape[1],)))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dense(512))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dense(256))
        self.model.add(Activation("relu"))
        self.model.add(BatchNormalization())
        self.model.add(Dense(output_shape))
        self.model.compile(optimizer=RMSprop(),
                           loss='mse')
        self.model.summary()

    def train(self, train_x, train_y, batch_size, nb_epoch, verbose=1):
        self.model.fit(train_x, train_y, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)

    def eval(self, test_x, test_y, verbose=0):
        score = self.model.evaluate(test_x, test_y, verbose=verbose)
        print(score)

    def predict(self, tk):
        pass

    def save(self, sdir, overwrite=True):
        self.model.save(os.path.join(sdir, "model.h5"))
        self.model.save_weights(os.path.join(sdir, "weights.h5"), overwrite=overwrite)

    def load(self, sdir):
        self.model = load_model(os.path.join(sdir, "model.h5"))
        self.model.load_weights(os.path.join(sdir, "weights.h5"))

if __name__ == "__main__":
    input_dim = 1
    
    output_dim = 1

    samples = 20

    x = np.random.random((samples, input_dim))
    y = np.random.random((samples, output_dim))
    md = Model()
    input_shape  = x.shape
    output_shape = 1
    md.create_model(input_shape, output_shape)
    md.train(x, y, 1, 1)
    """
    pred = cb.model.predict(p_x)
    print(pred, pred.shape)
    """
    gc.collect()
