"""
This script defines an agent that uses a Decision Tree Classifier to learn 
from data and make predictions about directional movements. 
The data is read from a CSV file, and the model is trained to predict 
movements based on the provided features. After training, the agent can 
predict the direction for new observations.
"""
import os
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib
import numpy as np



script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, 'decision_tree_model.joblib')

class AgentT:
    """
    An agent that learns to predict movement directions using a Decision Tree Classifier.
    """
    def __init__(self, data):
        """
        Initializes the agent with data.

        Parameters:
        - data: The input data for training.
        """
        self.data = data.drop_duplicates()



    def train(self):
        """
        Trains the Decision Tree Classifier on the provided data.
        """
        data = self.data
        data['target'] = data['target'].replace(
            { 
                'up':0,
                'down':1,
                'left':2,
                'right':3

            }
        )

        print(data[:10])
        X = data.drop('target', axis= 1)
 
        y = data['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

        dtc = DecisionTreeClassifier(criterion='entropy')
      
        dtc.fit(X_train, y_train)
        y_pred = dtc.predict(X_test)
        joblib.dump(dtc, 'decision_tree_model.joblib')
        print(confusion_matrix(y_test, y_pred))
        print(accuracy_score(y_test, y_pred))
        # print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_pred, y_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
    def test(self, observation):
        """
        Tests the model with a new observation and predicts the direction.

        Parameters:
        - observation: Input features for prediction.

        Returns:
        - predicted_direction: The predicted direction as a string.
        """
        direction_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        dtc = joblib.load('decision_tree_model.joblib')
        num_features = len(observation)  # Dynamic number of features
        columns = [f'f{i+1}' for i in range(num_features)]  # Create feature names dynamically

        # Create a DataFrame
        df = pd.DataFrame([observation], columns=columns)
        prediction = dtc.predict(df)
        predicted_direction = direction_map[prediction[0]]
        return predicted_direction



        