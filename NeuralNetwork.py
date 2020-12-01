from Layer import Layer
import json

def load_model(file_name):
    with open(file_name, 'r') as file:
        json_model = json.loads(file.read())
        model = NeuralNetwork(json_model['num_inputs'], len(json_model['hidden_layer']), len(json_model['output_layer']))
        model.set_params(json_model['learning_rate'], json_model['alpha'])
        hidden_layer = model.layers[0]
        output_layer = model.layers[1]
        for index, neuron in enumerate(hidden_layer.neurons):
            neuron.weights = {int(key): value for key,value in json_model['hidden_layer'][index].items()}
            neuron.bias = json_model['hidden_bias'][index]
        for index, neuron in enumerate(output_layer.neurons):
            neuron.weights = {int(key): value for key, value in json_model['output_layer'][index].items()}
            neuron.bias = json_model['output_bias'][index]
        return model
    return

class NeuralNetwork:
    def __init__(self, num_inputs, num_hidden_neurons, num_outputs):
        hidden_layer = Layer("Hidden Layer", num_inputs,num_hidden_neurons)
        output_layer = Layer("Output Layer", num_hidden_neurons, num_outputs)
        self.layers = [hidden_layer, output_layer]
        self.learning_rate = 0
        self.alpha = 0
        return

    def describe(self):
        print('Neural Network Description')
        print('Learning Rate: ', self.learning_rate)
        print('Alpha: ', self.alpha)
        print('=========================================================')
        for layer in self.layers:
            layer.describe()
            print('=========================================================')
        return

    def __forward_propagate(self,inpt):
        inputs = inpt
        for layer in self.layers:
            inputs = layer.activate(inputs)
        return inputs

    def __back_propagate(self, expected):
        hidden_layer = self.layers[0]
        output_layer = self.layers[1]
        output_layer.calc_output_deltas(expected)
        hidden_layer.calc_hidden_deltas(output_layer)
        return

    def __update_weights(self, inputs):
        hidden_layer = self.layers[0]
        output_layer = self.layers[1]
        hidden_outputs = hidden_layer.get_outputs()
        output_layer.update_weights(hidden_outputs, self.learning_rate, self.alpha)
        hidden_layer.update_weights(inputs, self.learning_rate, self.alpha)
        return

    def set_params(self, learning_rate, alpha):
        self.learning_rate = learning_rate
        self.alpha = alpha
        return

    def train(self, epochs, X, y, val_X, val_y):
        for epoch in range(epochs):
            train_error = 0.0
            for index, inpts in enumerate(X):
                outputs = self.__forward_propagate(inpts)
                train_error += sum([(y[index][i] - outputs[i])**2 for i in range(len(outputs))])
                self.__back_propagate(y[index])
                self.__update_weights(y[index])
            train_error = train_error / len(X)
            val_error = 0.0
            for index, inpt in enumerate(val_X):
                predicted_value = self.__forward_propagate(inpt)
                val_error += sum([(val_y[index][i] - predicted_value[i])**2 for i in range(len(predicted_value))])
            val_error = val_error / len(val_X)
            print("Epoch: %d, Training Error: %.3f Validation Error: %.3f" % (epoch+1, train_error, val_error))

    def predict(self, input):
        return self.__forward_propagate(input)
    
    def save(self, file_name):
        model = {}
        model['learning_rate'] = self.learning_rate
        model['alpha'] = self.alpha
        model['num_inputs'] = len(self.layers[0].neurons[0].weights)
        model['hidden_layer'] = [neuron.weights for neuron in self.layers[0].neurons]
        model['hidden_bias'] = [neuron.bias for neuron in self.layers[0].neurons]
        model['output_layer'] = [neuron.weights for neuron in self.layers[1].neurons]
        model['output_bias'] = [neuron.bias for neuron in self.layers[1].neurons]
        string_model = json.dumps(model)
        with open(file_name, 'w') as file:
            file.write(string_model)
        return