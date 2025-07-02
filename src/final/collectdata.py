"""
This script collects data from a series of games played using the GameLogic class. 
It simulates movements of an agent (like an ant) in a grid and records the features 
and actions taken during the game. The collected data is then saved in a CSV file for 
further use in training machine learning models like neural networks or decision trees.
"""
import os
import pandas as pd
from gamelogicF import GameLogic

GAMES_TO_PLAY = 10
GRID_LENGTH = 5
ANT_VIEW = 2

def main():
    """
    Main function to simulate games and collect training data.
    """
    games_to_play = GAMES_TO_PLAY
    games_played = 0
    step = 1
    train_data = []
    while games_played < games_to_play:
        grid_length = GRID_LENGTH 
        ant_view = ANT_VIEW
        game = GameLogic(grid_length,ant_view)
        solution = game.solution()
        print(solution)
        
        done = False
        print(f'game {games_played}, Solution {solution}')
    
        while not done:
            # print(f'pos {game.ant_position}')
            # print(f'visited \n {game.visited}')
            if len(solution) != 0:
                move = solution[0]
                neighbor = game.get_neighborhood()
                neighbor.append(move)
                train_data.append(neighbor)
                # print(f'{neighbor}')
                game.move(move)
                solution.pop(0)
                done = game.game_over()
                step += 1
            else :
                done = True
        games_played += 1

    num_features = (2 *game.m+ 1) **2 
    columns = [f'f{i+1}' for i in range(num_features)]  
    columns.append('target')  # Add the target column
    # Create a DataFrame
    df = pd.DataFrame(train_data, columns=columns)
    print(df)
    # Save to a CSV file
    # Print working directory and list files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    csv_path = os.path.join(data_dir, 'data_N' + str(game.N) +'_M' + str(game.m) + '_g'+ str(games_to_play) + '.csv')

    df.to_csv(csv_path, index=False)


if __name__ == '__main__':
    main()
   