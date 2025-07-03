from sklearn.linear_model import LinearRegression
import random
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import numpy as np
from gplearn.genetic import SymbolicRegressor



script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, 'model')
model_path = os.path.join(data_dir, 'geneticK_model.joblib')

    # Step 4: Interpret OUT
def interpret_output(out, K=1.0):


    if out <= -K:
        return 'up'
    elif out <= 0:
        return 'down'
    elif out <= K:
        return 'right'
    else:
        return 'left'

class AgentG:

    def __init__(self, data):

        self.data = data.drop_duplicates()

    def train(self):
        data = self.data
        data['target'] = data['target'].replace(
            { 
                'up': -2,
                'down': -0.5,
                'right': 0.5,
                'left': 2

            }
        )

        print(data[:10])
        X = data.drop('target', axis= 1)
 
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) #, random_state=41
        gp = SymbolicRegressor(
            population_size=200,
            generations=100,
            stopping_criteria=0.01,
            p_crossover=0.7,
            p_subtree_mutation=0.1,
            p_hoist_mutation=0.05,
            p_point_mutation=0.1,
            max_samples=0.9,
            verbose=1,
        )
       
        gp.fit(X_train, y_train)

        joblib.dump(gp, model_path)



   
    # Step 2: Add ERCs (e.g., 3 random constants per input)
    # def add_ercs(X_raw, num_ercs=3):
    #     X_augmented = []
    #     for row in X_raw:
    #         ercs = [random.uniform(-2, 2) for _ in range(num_ercs)]
    #         X_augmented.append(row + ercs)
    #     return X_augmented

    # X = add_ercs(X_raw, num_ercs=3)  # Now input dimension = 12






        
    def test(self, observation):
        model = joblib.load(model_path)
        num_features = len(observation)
        columns = [f'f{i+1}' for i in range(num_features)]
        df = pd.DataFrame([observation], columns=columns)
        prediction = model.predict(df)

        predicted_direction = interpret_output(prediction[0])
        return predicted_direction



