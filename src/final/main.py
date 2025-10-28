import os
import pandas as pd
from gamelogicF import GameLogic
from agentN import AgentN
import matplotlib.pyplot as plt
import numpy as np
from agentT import AgentT
from agentG import AgentG
# Get the path of the current script's directory

def barplot(scores, model_name):
    scores = scores
    games_to_play = len(scores)
    plt.figure(figsize=(12, 6))

    plt.bar(range(1, games_to_play+1 ), scores, color='b')
    plt.title("Scores of " + model_name + " Across" + str(games_to_play) + " Games")
    plt.xlabel("Game Number")
    plt.ylabel("Score")
    plt.xticks(range(5, games_to_play+1, 5), rotation=45)  # Set x-ticks to game numbers
    # plt.grid(True)
    # plt.grid(True, axis='y')
    plt.show()

def barchart(scores, model_name):
    scores = scores
    unique_scores, counts = np.unique(scores, return_counts=True)
    games_to_play = len(scores)
    plt.figure(figsize=(7, 7))  # Set figure size for better visualization
    plt.pie(counts, labels=unique_scores, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])

    plt.title("Distribution of Scores in " + model_name + " of " +str(games_to_play) + " Games")
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.

    # Show the pie chart
    plt.show()



def neural_network_agent(data, grid_length, ant_view):
    input_size = (2 * ant_view + 1) ** 2
    agent = AgentN(data, input_size, 8, 4)
    agent.train()

    games_to_play = 10
    scores = []
    for i in range(games_to_play):


        game = GameLogic(grid_length,ant_view)
        done = False
        while not done:
            neighbor = game.get_neighborhood()
            move = agent.test(neighbor)
            game.move(move)
            done = game.game_over()
            # print(f'move {move_decoded} reward = {game.reward}')
        scores.append(game.get_outcome())
        # print(f'Score {game.score}')
    return scores



def tree_agent(data, grid_length, ant_view):
    agent = AgentT(data)
    agent.train()

    
    games_to_play = 10
    scores = []
    for i in range(games_to_play):

        game = GameLogic(grid_length,ant_view)
        done = False
        while not done:
            neighbor = game.get_neighborhood()
            move = agent.test(neighbor)
            game.move(move)
            done = game.game_over()
            # print(f'move {move_decoded} reward = {game.reward}')
        scores.append(game.get_outcome())
        # print(f'Score {game.score}')
    return scores

def genetic_agent(data, grid_length, ant_view):
    agentg = AgentG(data)
    agentg.train()

    
    games_to_play = 10
    scores = []
    for i in range(games_to_play):

        game = GameLogic(grid_length,ant_view)
        done = False
        while not done:
            neighbor = game.get_neighborhood()
            move = agentg.test(neighbor)
            game.move(move)
            done = game.game_over()
            # print(f'move {move_decoded} reward = {game.reward}')
        scores.append(game.get_outcome())
        # print(f'Score {game.score}')
    return scores

def main():
    games_to_play = 10
    grid_length = 3
    ant_view = 1
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    csv_path = os.path.join(data_dir, 'data_N' + str(grid_length) +'_M' + str(ant_view) + '_g'+ str(games_to_play) + '.csv')
    data = pd.read_csv(csv_path)
    scores = neural_network_agent(data, grid_length, ant_view)
    # scores = tree_agent(data, grid_length, ant_view)
    # scores = genetic_agent(data, grid_length, ant_view)
 


    barplot(scores, "Tree")
    barchart(scores, "Tree")


        
if __name__ == '__main__':
    main()