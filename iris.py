from sklearn import datasets

iris = datasets.load_iris()

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np

X = iris.data
flower_name = iris.target
y = []

for i in flower_name:
    if i == 0:
        y.append([1, 0, 0])
    elif i == 1:
        y.append([0, 1, 0])
    elif i == 2:
        y.append([0, 0, 1])

print(y)

model = Sequential()
model.add(Dense(8, input_dim=4))
model.add(Activation('tanh'))
model.add(Dense(3))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.1)
model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=sgd)

model.fit(X, y, batch_size=30, nb_epoch=100)

answers = model.predict_proba(X)

for answer in answers:
    target_index = np.argmax(answer)
    print(iris.target_names[target_index])

print(
    model.predict_proba(
        [
            np.array([4.8, 3.4, 1.6, 0.2])
        ]
    )
)
