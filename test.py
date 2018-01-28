from keras.models import load_model
import dataset as dt
import numpy as np

fr, to = 0, 26

price, sol = dt.create_dataset(fr, to-1,'test.xlsx')
train_X = np.array(price)
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))

model = load_model('test2.h5')

p = model.predict(train_X)
print(p)
print(sol)
