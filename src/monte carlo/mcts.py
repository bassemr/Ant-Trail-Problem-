import random
import time
import math
import numpy as np
from copy import deepcopy
from gamelogic import GameLogic
from sortedcontainers import SortedDict
from gamegrphic import GameGrphic
import pickle
import pygame
import matplotlib.pyplot as plt
import pandas as pd
import os


class MCTSMeta:
    """
    Holds meta-parameters for Monte Carlo Tree Search (MCTS), including the exploration constant.
    """
    EXPLORATION = math.sqrt(2)

class Node:
    """
    Represents a node in the MCTS tree, storing information about moves, parent nodes, visit counts, and rewards.
    """
    def __init__(self, move, parent):
        self.move = move  # The move leading to this node
        self.parent = parent  # Reference to the parent node
        self.N = 0  # Number of times this node has been visited
        self.Q = 0  # Total value of this node
        self.R = 0  # Reward for this node
        self.children = SortedDict()  # Children of this node
        self.outcome = 0  # Outcome value

    def add_children(self, children: dict) : 
        """
        Adds child nodes to this node.
        """
        for child in children:
            self.children[child.move] = child

    def value(self, explore: float = MCTSMeta.EXPLORATION):
        """
        Calculates the node's value using the Upper Confidence Bound (UCB1) formula.
        """
        if self.N == 0:
            return np.inf

        else:
            return self.Q / self.N + explore * math.sqrt(math.log(self.parent.N) / self.N)
    def ucb1(self, C: float) -> float:
        """
        Computes the UCB1 value for this node.
        """
        if self.N == 0:
            return np.inf
        else:
            return (self.Q / self.N) + C * math.sqrt(math.log(self.parent.N + 1) / (self.N + 1))

class MCTS:

    """
    Main class for executing the Monte Carlo Tree Search algorithm, managing the search process and node expansion.
    """
    def __init__(self, state:GameLogic):
        self.root_state = self.serialize_state(state)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.state_map = {} 

    def serialize_state(self, state: GameLogic) -> bytes:
        """
        Serializes the game state into a binary format for storage.
        """
        return pickle.dumps(state)

    def deserialize_state(self, serialized_state: bytes) -> GameLogic:
        """
        Converts the serialized state back into a GameLogic object.
        """
        return pickle.loads(serialized_state)

    # Selecting
    def select_node(self) -> tuple:
        """
        Selects a node to explore based on the UCB1 algorithm and expands if necessary.
        """
        node = self.root
        state = self.deserialize_state(self.root_state)
       
        while len(node.children) != 0:
            children = node.children.values()
            C = MCTSMeta.EXPLORATION
            max_value = max(children, key=lambda n: n.value()).value()
            max_nodes = [n for n in children if n.value() == max_value]

            node = random.choice(max_nodes)
            state.move(node.move)
            node.R = state.get_reward()

            if node.N == 0:
                return node, self.serialize_state(state)

        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.move(node.move)
            node.R = state.get_reward()
        return node, self.serialize_state(state)

    def expand(self, parent: Node, state: GameLogic) -> bool:
        """
        Expands the current node by adding child nodes based on legal moves.
        """
        if state.game_over():
            return False

        children = [Node(move, parent) for move in state.get_legal_moves()]
        parent.add_children(children)

        return True
    def epsilon_greedy_move(self, state: GameLogic) -> str:
        """
        Chooses a move using an epsilon-greedy strategy for exploration and exploitation.
        """
        if random.random() < 0.1:
            # Explore: choose a random move
            return random.choice(state.get_legal_moves())
        else:
            DIRECTIONS = {
                'up': (-1, 0),
                'down': (1, 0),
                'left': (0, -1),
                'right': (0, 1)
            } 
            # Exploit: choose the best-known move
            legal_moves = state.get_legal_moves()
            best_move = None
            best_value = -np.inf
            best_moves = []
            for move in legal_moves:
                dx, dy = DIRECTIONS[move]
                # Evaluate the move by simulating it
                state_copy = self.deserialize_state(self.serialize_state(state))
                state_copy.move(move)
                nx = state_copy.ant_position[0] + dx
                ny = state_copy.ant_position[1] + dy
                reward = state_copy.get_reward()
                state_copy.visited, _ = state_copy.discover_new_cells(nx, ny, state_copy.visited)
                potential_discovery = np.count_nonzero(state_copy.visited == -np.inf)
                best_moves.append((move, reward, potential_discovery))
            best_moves.sort(key=lambda move: (-move[1], move[2]), reverse=False)
            
            return best_moves[0][0]

    def roll_out(self, serialized_state: bytes) -> int:
        """
        Performs a rollout from the given state until a terminal state is reached, returning the outcome.
        """
        state = self.deserialize_state(serialized_state)
        while not state.game_over():
            move = self.epsilon_greedy_move(state)
            state.move(move)

        return state.get_outcome() + 0.1 * state.moves_left 


    def evaluate_move(self, state: GameLogic) -> float:
        # Evaluate a move based on some criteria, e.g., current score or heuristic
        return state.get_reward()  # 

    def back_propagate(self, node: Node, outcome: int) -> None:
        """
        Backpropagates the outcome through the tree, updating visit counts and total values.
        """

        while node is not None:
            node.N += 1
            node.Q += outcome
            node = node.parent



    def search(self, time_limit: int):
        """
        Executes the MCTS search process for a specified time limit.
        """
        start_time = time.process_time()
        num_rollouts = 0
        while time.process_time() - start_time < time_limit:
            node, serialized_state = self.select_node()
            outcome = self.roll_out(serialized_state)
            self.back_propagate(node, outcome)
            num_rollouts += 1

        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.num_rollouts = num_rollouts



    def best_move(self):
        """
        Determines the best move based on the number of visits to child nodes.
        """
        current_state = self.deserialize_state(self.root_state)

        if current_state.game_over():
            return -1

        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)
        return best_child.move

    def move(self, move):
        """
        Updates the current state and tree based on the chosen move.
        """
        current_state = self.deserialize_state(self.root_state)
        if move in self.root.children:
            current_state.move(move)
            self.root_state = self.serialize_state(current_state)
            self.root = self.root.children[move]
        else:
            current_state.move(move)
            self.root_state = self.serialize_state(current_state)
            self.root = Node(None, None)

    def statistics(self) -> tuple:
        """
        Returns statistics about the search process, including rollouts and run time.
        """
        return self.num_rollouts, self.run_time
    


def create_one_game():
    """
    Initializes and runs a single game instance with graphical output.
    """
    state = GameLogic(3,1)
    gamegraph = GameGrphic(state)
    mcts = MCTS(state)
    print(state.grid)
    gamegraph.draw()
    running = True
    while not state.game_over():
        for event in pygame.event.get():
            if event.type == pygame.QUIT :
                # running = False
                pygame.quit()
                quit()
        # print(game.grid)
        
        mcts.search(0.1)
        num_rollouts, run_time = mcts.statistics()
        print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
        move = mcts.best_move()
        mcts.move(move)
        print("MCTS chose move: ", move)
        # print(moves)
        # move = np.random.choice(moves)
        gamegraph.draw_direction(move)
        state.move(move)
        gamegraph.draw()
    print(f'{state.score=}')

def create_multiple_games(games_to_play, N, m, create_data):
    """
    Runs multiple games and records the scores.
    """
    games_to_play = games_to_play
    scores = []
    train_data = []
    for i in range(games_to_play):
        state = GameLogic(N,m)
        mcts = MCTS(state)
        # print(state.grid)
        running = True
        while not state.game_over():
        
            # print(game.grid)
            
            mcts.search(0.1)
            num_rollouts, run_time = mcts.statistics()
            # print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
            move = mcts.best_move()
            neighbor = state.get_neighborhood()
            neighbor.append(move)
            train_data.append(neighbor)
            mcts.move(move)
            # print("MCTS chose move: ", move)
            # print(moves)
            # move = np.random.choice(moves)
            state.move(move)
        scores.append(state.score)
    if(create_data):
        num_features = (2 *m+ 1) **2 
        columns = [f'f{i+1}' for i in range(num_features)]  
        columns.append('target')
        df = pd.DataFrame(train_data, columns=columns)
        # print(df)
        # Save to a CSV file
        # Print working directory and list files
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, '..', 'final', 'data')
        data_dir = os.path.abspath(data_dir)  # Normalize the path
        csv_path = os.path.join(data_dir, 'data_N' + str(N) +'_M' + str(m) + '_g'+ str(games_to_play) + '.csv')
        df.to_csv(csv_path, index=False)





    return scores


def barplot(scores):
    """
    Generates a bar plot of scores across multiple games.
    """
    scores = scores
    games_to_play = len(scores)
    plt.figure(figsize=(12, 6))

    plt.bar(range(1, games_to_play+1 ), scores, color='b')
    plt.title("Scores of MCTS Across " +str(games_to_play) + " Games")
    plt.xlabel("Game Number")
    plt.ylabel("Score")
    plt.xticks(range(5, games_to_play+1, 5), rotation=45)  # Set x-ticks to game numbers
    # plt.grid(True)
    # plt.grid(True, axis='y')
    plt.show()

def barchart(scores):
    """
    Generates a pie chart showing the distribution of scores across games.
    """
    scores = scores
    unique_scores, counts = np.unique(scores, return_counts=True)
    games_to_play = len(scores)
    plt.figure(figsize=(7, 7))  # Set figure size for better visualization
    plt.pie(counts, labels=unique_scores, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen', 'lightcoral'])

    plt.title("Distribution of Scores in MCTS of " +str(games_to_play) + " Games")
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.

    # Show the pie chart
    plt.show()




 

def main(once=True, games_to_play=10):
    N = 10
    m = 2
    create_data = False
    """
    Main entry point for running the program, either a single game or multiple games.
    """
    if once:
        scores = create_multiple_games(games_to_play, N, m, create_data)
        barplot(scores)
        barchart(scores)

    else:
        create_one_game()




if __name__ == '__main__':
    main()