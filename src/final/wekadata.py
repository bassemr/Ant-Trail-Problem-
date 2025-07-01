"""
This script is used to generate training and testing data for a machine learning model,
which will be formatted in ARFF (Attribute-Relation File Format) for use in Weka,
a popular data mining software. The data is collected by simulating games and capturing the states and actions taken by the agent in the game.
"""

import os
import pandas as pd
from gamelogicF import GameLogic
from sklearn.model_selection import train_test_split


def write_arff(df, filename, relation_name):
    """
    Writes a DataFrame to an ARFF file format, including relation and attribute definitions.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    arff_path = os.path.join(script_dir, filename + '.arff')
    with open(arff_path, 'w') as f:
        f.write(f"@relation {relation_name}\n\n")

        print(df.columns)
        # Write the attributes
        for column in df.columns[:-1]:

                f.write(f"@attribute {column} numeric \n")
            
        f.write(f"@attribute {df.columns[-1]} {{U, D, L, R}} \n")
        
        f.write("\n@data\n")
        
        # Write the data
        for i, row in enumerate(df.iterrows()):
            row_str = ','.join(map(str, row[1].values))
            if i < len(df) - 1:
                f.write(f"{row_str}\n")
            else:
                f.write(f"{row_str}")
    


def main():
    """
    Main function to simulate games and collect data for training a model.
    """
    grid_length = 3
    ant_view = 1
    games_to_play = 100
    games_played = 0
    step = 1
    train_data = []
    while games_played < games_to_play:

        game = GameLogic(grid_length, ant_view)
        solution = game.solution()
        done = False
        # print(f'game {games_played}, Solution {solution}')
    
        while not done:
            # print(f'pos {game.ant_position}')
            # print(f'visited \n {game.visited}')
            if len(solution) != 0:
                move = solution[0]
                neighbor = game.get_neighborhood()
                neighbor.append(move[0].upper())
                train_data.append(neighbor)
                # print(f'{neighbor}')
                game.move(move)
                solution.pop(0)
                done = game.game_over()
                step += 1
            else :
                done = True
        games_played += 1

    num_features = (2 *game.m+ 1) **2  # Dynamic number of features
    columns = [f'f{i+1}' for i in range(num_features)]  # Create feature names dynamically
    columns.append('target')  # Add the target column
    # Create a DataFrame
    df = pd.DataFrame(train_data, columns=columns)
    df = df.drop_duplicates()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    file_name = str(grid_length) + '_' + str(ant_view)
    write_arff(train_df, 'ant' + file_name + 'train', 'ant'+ file_name  )
    write_arff(test_df, 'ant' + file_name +'test', 'ant' + file_name  )


if __name__ == '__main__':
    main()
   