from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM, Bidirectional
from keras.optimizers import RMSprop, Nadam
from matplotlib import pyplot
import dataset as dt
import numpy as np

to = 300000
dropout_value = 0.2
perc = int(to*0.8)
window_size = 10
price, sol = dt.create_dataset(0, to-1,'1mgdax.xlsx')

train_X = np.array(price[:perc])
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = np.array(price[perc:])
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
train_y = sol[:perc, :]
train_y = train_y
test_y = sol[perc:, :]
test_y = test_y

model = Sequential()
model.add(LSTM(window_size * 10, return_sequences=True, input_shape=(1, train_X.shape[-1])))
model.add(Dropout(dropout_value))
model.add(Dense(units=window_size * 20, activation='relu', use_bias=True))
model.add(Dropout(dropout_value))
model.add(Dense(units=window_size * 15, activation='relu', use_bias=True))
model.add(Dropout(dropout_value))
model.add(LSTM((window_size * 10), return_sequences=True))
model.add(Dropout(dropout_value))
model.add(LSTM(window_size * 5, return_sequences=False))
model.add(Dense(units=window_size * 2, activation='relu', use_bias=True))
model.add(Dense(units=2, activation='sigmoid', use_bias=True))

#sgd = RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
sgd = Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(loss='binary_crossentropy', optimizer=sgd)
history = model.fit(train_X, train_y,
                    epochs=2000, batch_size=1000,
                    validation_data=(test_X, test_y),
                    verbose=1, shuffle=False)

model.save('test2.h5')

