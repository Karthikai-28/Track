import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

np.random.seed()

n_pts = 500
X, y = datasets.make_circles(n_samples=n_pts, noise=0.1, random_state=None, factor=0.2)

plt.scatter(X[y==0, 0], X[y==0, 1])
plt.scatter(X[y==1, 0], X[y==1, 1])

model = Sequential()
model.add(Dense(4, input_shape=(2,), activation="sigmoid"))
model.add(Dense(1, activation="sigmoid"))
model.compile(Adam(learning_rate=0.01), 'binary_crossentropy', metrics = ["accuracy"])

h = model.fit(x=X, y=y, batch_size=20, epochs=100, verbose=1, shuffle=True)

plt.plot(h.history['accuracy'])
plt.xlabel('epoch')
plt.legend('accuracy')
plt.title('accuracy')

plt.plot(h.history['loss'])
plt.xlabel('epoch')
plt.legend('loss')
plt.title('loss')

def plot_decision_boundary(X, y, model):
  x_span = np.linspace(min(X[:, 0]) - 0.25, max(X[:, 0]) + 0.25)
  y_span = np.linspace(min(X[:, 1]) - 0.25, max(X[:, 1]) + 0.25)
  xx, yy = np.meshgrid(x_span, y_span)
  xx_, yy_ = xx.ravel(), yy.ravel()
  grid = np.c_[xx_, yy_]
  predict_function = model.predict(grid)
  z = predict_function.reshape(xx.shape)
  plt.contourf(xx, yy, z)

plot_decision_boundary(X, y, model)
plt.scatter(X[:n_pts, 0], X[:n_pts, 1])
plt.scatter(X[n_pts:, 0], X[n_pts:, 1])
x = 0.1
y = 0.75
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker = 'o', markersize = 10, color = 'red')
print("Prediction is: ", prediction)
