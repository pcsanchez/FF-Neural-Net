from NeuralNetwork import NeuralNetwork
from NeuralNetwork import load_model
from utils import read_csv


# X = [[0, 0], [1, 0], [0, 1], [1, 1]]

# y = [[0], [0], [0], [1]]

# val_X = [[1,1]]
# val_y = [[1]]

# model = NeuralNetwork(1,2,1)
# model.set_params(0.5, 0.7)
# model.save('model.json')
# model.describe()

model = load_model('model.json')
model.describe()
# model.train(10000, X, y, val_X, val_y)
# print(model.predict([0, 0]))
# print(model.predict([1, 0]))
# print(model.predict([0, 1]))
# print(model.predict([1, 1]))

