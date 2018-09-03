import math
import random

class NNetwork(object):
    biases = []
    activations = []
    weights = []
    deltas = []

    def __init__(self, inputs_count, hiddent_layer_count, output_count):
        self.inputs = inputs_count
        self.hidden = hiddent_layer_count
        self.output = output_count

        #set weights for hidden layer
        for j in range(self.inputs):
            neuron_weights = []
            for i in range(self.hidden):
                neuron_weights.append(random.random())
            self.weights.append(neuron_weights)

        #set weights for output layer
        for j in range(self.hidden):
            neuron_weights = []
            for i in range(self.output):
                neuron_weights.append(random.random())
            self.weights.append(neuron_weights)

        #set stuff
        for i in range(self.hidden + self.output):
            self.biases.append(random.random())
            self.activations.append(0)
            self.deltas.append(0)

    def feedforward(self, input_layer, show_result = False):
        for neuronNum in range(self.hidden):
            sum = 0
            for i in range(self.inputs):
                sum += self.weights[i][neuronNum] * input_layer[i]
            sum += self.biases[neuronNum]
            self.activations[neuronNum] = sigmoid(sum)

        for last_layer_neuron in range(self.output):
            sum = 0
            for i in range(self.hidden):
                sum += self.weights[self.inputs+i][last_layer_neuron] * self.activations[i]
            sum += self.biases[last_layer_neuron]
            self.activations[self.hidden+last_layer_neuron] = sigmoid(sum)

        if(show_result):
            print("{0}; {1}; {2}; {3} => {4} ({7}) - {5} ({8}) - {6} ({9})"
            .format(round(input_layer[0],3),round(input_layer[1],3),round(input_layer[2],3),round(input_layer[3],3),
            round(self.activations[self.hidden],3),round(self.activations[self.hidden+1],3),round(self.activations[self.hidden+2],3),
            input_layer[4],input_layer[5],input_layer[6]))

    def learnNN(self, learning_data, epoch, learning_rate, learning_accuracy):
        max_epoch = epoch
        epoch = 0
        mse = 100
        learning_data_len = len(learning_data)
        # Shuffle data
        for i in range(learning_data_len):
            k = random.randint(i,learning_data_len-1)
            tmp = learning_data[i]
            learning_data[i] = learning_data[k]
            learning_data[k] = tmp

        while epoch < max_epoch and mse > learning_accuracy:
            for input in learning_data:
                self.feedforward(input)
                #traning network
                for i in range(self.output):
                    gradient = self.activations[self.hidden+i] * (1-self.activations[self.hidden+i])
                    self.deltas[self.hidden+i] = (input[self.inputs + i] - self.activations[self.hidden+i]) * gradient

                for neuron_index in range(self.hidden):
                    gradient = self.activations[neuron_index] * (1 - self.activations[neuron_index])
                    sum = 0
                    for i in range(self.output):
                        sum += self.weights[self.inputs + neuron_index][i] * self.deltas[self.hidden + i]
                    self.deltas[neuron_index] = gradient * sum

                # weights adjusting
                for i in range(self.inputs):
                    for neuron_index in range(self.hidden):
                        learning_delta = learning_rate * self.deltas[neuron_index] * input[i]
                        self.weights[i][neuron_index] += learning_delta
                    
                for neuron_index in range(self.hidden):
                    self.biases[neuron_index] += learning_rate * self.deltas[neuron_index] * 1

                for j in range(self.hidden):
                    for i in range(self.output):
                        learning_delta = learning_rate * self.deltas[self.hidden + i] * self.activations[j]
                        self.weights[self.inputs + j][i] += learning_delta

                for neuron_index in range(self.output):
                    self.biases[self.hidden + neuron_index] += learning_rate * self.deltas[self.hidden + neuron_index] * 1

            epoch+=1
            mse = self.meanSqrError(learning_data)
            if epoch % 100 == 0:
                print('epoch {0}/{2} limit reached. MSE = {1}'.format(epoch,mse,max_epoch))
                self.feedforward(learning_data[0], True)
        print('DONE...')

    def meanSqrError(self, data):
        sum = 0
        for record in data:
            self.feedforward(record)
            for i in range(self.output):
                sum += math.pow((self.activations[self.hidden + i] - record[self.inputs+i]),2)
        return sum / len(data)

def sigmoid(x):
    return 1.0/(1.0 + math.exp(-x))

def accuracy_calc(dataset, nn, print_result = False):
    right_answ = 0
    wrong_answ = 0
    for row in dataset:
        nn.feedforward(row, print_result)
        max = -1
        k = -1
        for i in range(nn.output):
            if(max < nn.activations[nn.hidden+i]):
               k+=1
               max = nn.activations[nn.hidden+i]
        target = 0
        for i in range(nn.output):
            if(row[nn.inputs+i] == 1):
                target = i
        if(k == target):
            right_answ += 1
        else:
            wrong_answ += 1
    return right_answ/len(dataset)



if __name__ == '__main__':

    # Iris flower data set.
    # Data is sepal length, sepal width, petal length, petal width identifies iris species
    # Iris virginica  = 1 0 0;
    # Iris versicolor = 0 1 0;
    # Iris setosa     = 0 0 1 
    data = list()
    learning_data = list()
    test_data = list()
                #sepal length, width, petal length, width
    data.append([ 5.1         , 3.5  , 1.4         , 0.2  , 0, 0, 1 ])
    data.append([ 4.9, 3.0, 1.4, 0.2, 0, 0, 1 ])
    data.append([ 4.7, 3.2, 1.3, 0.2, 0, 0, 1 ])
    data.append([ 4.6, 3.1, 1.5, 0.2, 0, 0, 1 ])
    data.append([ 5.0, 3.6, 1.4, 0.2, 0, 0, 1 ])
    data.append([ 5.4, 3.9, 1.7, 0.4, 0, 0, 1 ])
    data.append([ 4.6, 3.4, 1.4, 0.3, 0, 0, 1 ])
    data.append([ 5.0, 3.4, 1.5, 0.2, 0, 0, 1 ])
    data.append([ 4.4, 2.9, 1.4, 0.2, 0, 0, 1 ])
    data.append([ 4.9, 3.1, 1.5, 0.1, 0, 0, 1 ])

    data.append([ 5.4, 3.7, 1.5, 0.2, 0, 0, 1 ])
    data.append([ 4.8, 3.4, 1.6, 0.2, 0, 0, 1 ])
    data.append([ 4.8, 3.0, 1.4, 0.1, 0, 0, 1 ])
    data.append([ 4.3, 3.0, 1.1, 0.1, 0, 0, 1 ])
    data.append([ 5.8, 4.0, 1.2, 0.2, 0, 0, 1 ])
    data.append([ 5.7, 4.4, 1.5, 0.4, 0, 0, 1 ])
    data.append([ 5.4, 3.9, 1.3, 0.4, 0, 0, 1 ])
    data.append([ 5.1, 3.5, 1.4, 0.3, 0, 0, 1 ])
    data.append([ 5.7, 3.8, 1.7, 0.3, 0, 0, 1 ])
    data.append([ 5.1, 3.8, 1.5, 0.3, 0, 0, 1 ])

    data.append([ 5.4, 3.4, 1.7, 0.2, 0, 0, 1 ])
    data.append([ 5.1, 3.7, 1.5, 0.4, 0, 0, 1 ])
    data.append([ 4.6, 3.6, 1.0, 0.2, 0, 0, 1 ])
    data.append([ 5.1, 3.3, 1.7, 0.5, 0, 0, 1 ])
    data.append([ 4.8, 3.4, 1.9, 0.2, 0, 0, 1 ])
    data.append([ 5.0, 3.0, 1.6, 0.2, 0, 0, 1 ])
    data.append([ 5.0, 3.4, 1.6, 0.4, 0, 0, 1 ])
    data.append([ 5.2, 3.5, 1.5, 0.2, 0, 0, 1 ])
    data.append([ 5.2, 3.4, 1.4, 0.2, 0, 0, 1 ])
    data.append([ 4.7, 3.2, 1.6, 0.2, 0, 0, 1 ])

    data.append([ 4.8, 3.1, 1.6, 0.2, 0, 0, 1 ])
    data.append([ 5.4, 3.4, 1.5, 0.4, 0, 0, 1 ])
    data.append([ 5.2, 4.1, 1.5, 0.1, 0, 0, 1 ])
    data.append([ 5.5, 4.2, 1.4, 0.2, 0, 0, 1 ])
    data.append([ 4.9, 3.1, 1.5, 0.1, 0, 0, 1 ])
    data.append([ 5.0, 3.2, 1.2, 0.2, 0, 0, 1 ])
    data.append([ 5.5, 3.5, 1.3, 0.2, 0, 0, 1 ])
    data.append([ 4.9, 3.1, 1.5, 0.1, 0, 0, 1 ])
    data.append([ 4.4, 3.0, 1.3, 0.2, 0, 0, 1 ])
    data.append([ 5.1, 3.4, 1.5, 0.2, 0, 0, 1 ])

    data.append([ 5.0, 3.5, 1.3, 0.3, 0, 0, 1 ])
    data.append([ 4.5, 2.3, 1.3, 0.3, 0, 0, 1 ])
    data.append([ 4.4, 3.2, 1.3, 0.2, 0, 0, 1 ])
    data.append([ 5.0, 3.5, 1.6, 0.6, 0, 0, 1 ])
    data.append([ 5.1, 3.8, 1.9, 0.4, 0, 0, 1 ])
    data.append([ 4.8, 3.0, 1.4, 0.3, 0, 0, 1 ])
    data.append([ 5.1, 3.8, 1.6, 0.2, 0, 0, 1 ])
    data.append([ 4.6, 3.2, 1.4, 0.2, 0, 0, 1 ])
    data.append([ 5.3, 3.7, 1.5, 0.2, 0, 0, 1 ])
    data.append([ 5.0, 3.3, 1.4, 0.2, 0, 0, 1 ])

    data.append([ 7.0, 3.2, 4.7, 1.4, 0, 1, 0 ])
    data.append([ 6.4, 3.2, 4.5, 1.5, 0, 1, 0 ])
    data.append([ 6.9, 3.1, 4.9, 1.5, 0, 1, 0 ])
    data.append([ 5.5, 2.3, 4.0, 1.3, 0, 1, 0 ])
    data.append([ 6.5, 2.8, 4.6, 1.5, 0, 1, 0 ])
    data.append([ 5.7, 2.8, 4.5, 1.3, 0, 1, 0 ])
    data.append([ 6.3, 3.3, 4.7, 1.6, 0, 1, 0 ])
    data.append([ 4.9, 2.4, 3.3, 1.0, 0, 1, 0 ])
    data.append([ 6.6, 2.9, 4.6, 1.3, 0, 1, 0 ])
    data.append([ 5.2, 2.7, 3.9, 1.4, 0, 1, 0 ])

    data.append([ 5.0, 2.0, 3.5, 1.0, 0, 1, 0 ])
    data.append([ 5.9, 3.0, 4.2, 1.5, 0, 1, 0 ])
    data.append([ 6.0, 2.2, 4.0, 1.0, 0, 1, 0 ])
    data.append([ 6.1, 2.9, 4.7, 1.4, 0, 1, 0 ])
    data.append([ 5.6, 2.9, 3.6, 1.3, 0, 1, 0 ])
    data.append([ 6.7, 3.1, 4.4, 1.4, 0, 1, 0 ])
    data.append([ 5.6, 3.0, 4.5, 1.5, 0, 1, 0 ])
    data.append([ 5.8, 2.7, 4.1, 1.0, 0, 1, 0 ])
    data.append([ 6.2, 2.2, 4.5, 1.5, 0, 1, 0 ])
    data.append([ 5.6, 2.5, 3.9, 1.1, 0, 1, 0 ])

    data.append([ 5.9, 3.2, 4.8, 1.8, 0, 1, 0 ])
    data.append([ 6.1, 2.8, 4.0, 1.3, 0, 1, 0 ])
    data.append([ 6.3, 2.5, 4.9, 1.5, 0, 1, 0 ])
    data.append([ 6.1, 2.8, 4.7, 1.2, 0, 1, 0 ])
    data.append([ 6.4, 2.9, 4.3, 1.3, 0, 1, 0 ])
    data.append([ 6.6, 3.0, 4.4, 1.4, 0, 1, 0 ])
    data.append([ 6.8, 2.8, 4.8, 1.4, 0, 1, 0 ])
    data.append([ 6.7, 3.0, 5.0, 1.7, 0, 1, 0 ])
    data.append([ 6.0, 2.9, 4.5, 1.5, 0, 1, 0 ])
    data.append([ 5.7, 2.6, 3.5, 1.0, 0, 1, 0 ])

    data.append([ 5.5, 2.4, 3.8, 1.1, 0, 1, 0 ])
    data.append([ 5.5, 2.4, 3.7, 1.0, 0, 1, 0 ])
    data.append([ 5.8, 2.7, 3.9, 1.2, 0, 1, 0 ])
    data.append([ 6.0, 2.7, 5.1, 1.6, 0, 1, 0 ])
    data.append([ 5.4, 3.0, 4.5, 1.5, 0, 1, 0 ])
    data.append([ 6.0, 3.4, 4.5, 1.6, 0, 1, 0 ])
    data.append([ 6.7, 3.1, 4.7, 1.5, 0, 1, 0 ])
    data.append([ 6.3, 2.3, 4.4, 1.3, 0, 1, 0 ])
    data.append([ 5.6, 3.0, 4.1, 1.3, 0, 1, 0 ])
    data.append([ 5.5, 2.5, 4.0, 1.3, 0, 1, 0 ])

    data.append([ 5.5, 2.6, 4.4, 1.2, 0, 1, 0 ])
    data.append([ 6.1, 3.0, 4.6, 1.4, 0, 1, 0 ])
    data.append([ 5.8, 2.6, 4.0, 1.2, 0, 1, 0 ])
    data.append([ 5.0, 2.3, 3.3, 1.0, 0, 1, 0 ])
    data.append([ 5.6, 2.7, 4.2, 1.3, 0, 1, 0 ])
    data.append([ 5.7, 3.0, 4.2, 1.2, 0, 1, 0 ])
    data.append([ 5.7, 2.9, 4.2, 1.3, 0, 1, 0 ])
    data.append([ 6.2, 2.9, 4.3, 1.3, 0, 1, 0 ])
    data.append([ 5.1, 2.5, 3.0, 1.1, 0, 1, 0 ])
    data.append([ 5.7, 2.8, 4.1, 1.3, 0, 1, 0 ])

    data.append([ 6.3, 3.3, 6.0, 2.5, 1, 0, 0 ])
    data.append([ 5.8, 2.7, 5.1, 1.9, 1, 0, 0 ])
    data.append([ 7.1, 3.0, 5.9, 2.1, 1, 0, 0 ])
    data.append([ 6.3, 2.9, 5.6, 1.8, 1, 0, 0 ])
    data.append([ 6.5, 3.0, 5.8, 2.2, 1, 0, 0 ])
    data.append([ 7.6, 3.0, 6.6, 2.1, 1, 0, 0 ])
    data.append([ 4.9, 2.5, 4.5, 1.7, 1, 0, 0 ])
    data.append([ 7.3, 2.9, 6.3, 1.8, 1, 0, 0 ])
    data.append([ 6.7, 2.5, 5.8, 1.8, 1, 0, 0 ])
    data.append([ 7.2, 3.6, 6.1, 2.5, 1, 0, 0 ])

    data.append([ 6.5, 3.2, 5.1, 2.0, 1, 0, 0 ])
    data.append([ 6.4, 2.7, 5.3, 1.9, 1, 0, 0 ])
    data.append([ 6.8, 3.0, 5.5, 2.1, 1, 0, 0 ])
    data.append([ 5.7, 2.5, 5.0, 2.0, 1, 0, 0 ])
    data.append([ 5.8, 2.8, 5.1, 2.4, 1, 0, 0 ])
    data.append([ 6.4, 3.2, 5.3, 2.3, 1, 0, 0 ])
    data.append([ 6.5, 3.0, 5.5, 1.8, 1, 0, 0 ])
    data.append([ 7.7, 3.8, 6.7, 2.2, 1, 0, 0 ])
    data.append([ 7.7, 2.6, 6.9, 2.3, 1, 0, 0 ])
    data.append([ 6.0, 2.2, 5.0, 1.5, 1, 0, 0 ])

    data.append([ 6.9, 3.2, 5.7, 2.3, 1, 0, 0 ])
    data.append([ 5.6, 2.8, 4.9, 2.0, 1, 0, 0 ])
    data.append([ 7.7, 2.8, 6.7, 2.0, 1, 0, 0 ])
    data.append([ 6.3, 2.7, 4.9, 1.8, 1, 0, 0 ])
    data.append([ 6.7, 3.3, 5.7, 2.1, 1, 0, 0 ])
    data.append([ 7.2, 3.2, 6.0, 1.8, 1, 0, 0 ])
    data.append([ 6.2, 2.8, 4.8, 1.8, 1, 0, 0 ])
    data.append([ 6.1, 3.0, 4.9, 1.8, 1, 0, 0 ])
    data.append([ 6.4, 2.8, 5.6, 2.1, 1, 0, 0 ])
    data.append([ 7.2, 3.0, 5.8, 1.6, 1, 0, 0 ])

    data.append([ 7.4, 2.8, 6.1, 1.9, 1, 0, 0 ])
    data.append([ 7.9, 3.8, 6.4, 2.0, 1, 0, 0 ])
    data.append([ 6.4, 2.8, 5.6, 2.2, 1, 0, 0 ])
    data.append([ 6.3, 2.8, 5.1, 1.5, 1, 0, 0 ])
    data.append([ 6.1, 2.6, 5.6, 1.4, 1, 0, 0 ])
    data.append([ 7.7, 3.0, 6.1, 2.3, 1, 0, 0 ])
    data.append([ 6.3, 3.4, 5.6, 2.4, 1, 0, 0 ])
    data.append([ 6.4, 3.1, 5.5, 1.8, 1, 0, 0 ])
    data.append([ 6.0, 3.0, 4.8, 1.8, 1, 0, 0 ])
    data.append([ 6.9, 3.1, 5.4, 2.1, 1, 0, 0 ])

    data.append([ 6.7, 3.1, 5.6, 2.4, 1, 0, 0 ])
    data.append([ 6.9, 3.1, 5.1, 2.3, 1, 0, 0 ])
    data.append([ 5.8, 2.7, 5.1, 1.9, 1, 0, 0 ])
    data.append([ 6.8, 3.2, 5.9, 2.3, 1, 0, 0 ])
    data.append([ 6.7, 3.3, 5.7, 2.5, 1, 0, 0 ])
    data.append([ 6.7, 3.0, 5.2, 2.3, 1, 0, 0 ])
    data.append([ 6.3, 2.5, 5.0, 1.9, 1, 0, 0 ])
    data.append([ 6.5, 3.0, 5.2, 2.0, 1, 0, 0 ])
    data.append([ 6.2, 3.4, 5.4, 2.3, 1, 0, 0 ])
    data.append([ 5.9, 3.0, 5.1, 1.8, 1, 0, 0 ])

    #getting learning data and test data
    for data_row in data:
        if(random.randint(0,100) > 70):
            test_data.append(data_row)
        else:
            learning_data.append(data_row)

    print("Initializing neural network...")
    nn = NNetwork(4,7,3)
    print("Starting neural network learning process...")
    nn.learnNN(learning_data, 2555, 0.0555, 0.05)

    print("Results:")
    test_accurance = accuracy_calc(test_data,nn,True)
    print("Learning data accuracy = {0}".format(round(accuracy_calc(learning_data,nn),3)))
    print("Test data accuracy     = {0}".format(round(test_accurance,3)))
    
