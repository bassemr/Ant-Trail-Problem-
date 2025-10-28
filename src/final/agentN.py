"""
This script defines an agent that uses a neural network model to learn from data
and make predictions based on the input features. 
The data is read from a CSV file, and the model is trained to predict directional 
movements based on the provided features. After training, the agent can be used to 
make predictions based on new observations.
"""
import torch.nn.grad
from  modelN import Model
import torch
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
import torch.nn as nn
import matplotlib.pyplot as plt

torch.manual_seed(41)



script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'data.csv')

class AgentN:
    """
    An agent that learns to predict movement directions using a neural network.
    """
    def __init__(self, data, input, h1, output=4):
        """
        Initializes the agent with data and model parameters.
        
        Parameters:
        - data: The input data for training.
        - input: Number of input features.
        - h1: Size of the hidden layer.
        - output: Number of output classes (default is 4 for directions).
        """
        self.data = data
        print(data)
        self.input_size = input
        self.hidden_layer_size =  h1
        self.output_size = output
        self.model = Model(self.input_size, self.hidden_layer_size, self.output_size)
        self.data = data.drop_duplicates()




    def train(self):
        """
        Trains the neural network model on the provided data.
        """

        data = self.data
        data.loc[data['target'] == 'up', 'target'] = 0
        data.loc[data['target'] == 'down', 'target'] = 1
        data.loc[data['target'] == 'left', 'target'] = 2
        data.loc[data['target'] == 'right', 'target'] = 3

        X = data.drop('target', axis= 1)
 
        y = data['target']

        X = X.values
        y = y.values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

        # Convert inputs and outputs to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        # Convert targets to LongTensor for classification (since CrossEntropyLoss expects this type)
        y_train = torch.LongTensor(y_train)
        y_test = torch.LongTensor(y_test)


        # Set the criterion of model to measure the error, how far off the predictions are from the data
        criterion = nn.CrossEntropyLoss()
        # Choose Adam Optimizer, lr = learning rate (if error doesn't go down after a bunch of iterations (epochs), lower our learning rate)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)



        epochs = 300
        losses = []
        for i in range(epochs):
            y_pred = self.model.forward(X_train)

            loss = criterion(y_pred, y_train)

            losses.append(loss.detach().numpy())
            if i % 10 == 0:
                print(f'Epoch {i} Loss {loss}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()



        # Graph it out!
        plt.plot(range(epochs), losses)
        plt.ylabel("loss/error")
        plt.xlabel('Epoch')
        plt.show(block=True)



        with torch.no_grad():
            y_eval = self.model.forward(X_test)
            y_pred = torch.argmax(y_eval, dim=1)
            loss = criterion(y_eval, y_test)
        correct_predictions = (y_pred == y_test).sum().item()  # Count correct predictions
        accuracy = correct_predictions / len(y_test) * 100  # Convert to percentage
        print(f'Accuracy: {accuracy:.2f}%')
        print(f' loss {loss}')

    def test(self, observation:list):
        """
        Tests the model with a new observation and predicts the direction.

        Parameters:
        - observation: Input features for prediction.

        Returns:
        - predicted_direction: The predicted direction as a string.
        """
        obs =  torch.FloatTensor(observation)
        direction_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        with torch.no_grad():
            obs = self.model(obs)
            idx = torch.argmax(obs, dim=0).item()
            predicted_direction = direction_map[idx]

        return predicted_direction




        