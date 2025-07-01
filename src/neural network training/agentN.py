import torch
import random
import matplotlib.pyplot as plt
import pygame
import numpy as np 
from collections import deque
from gamelogicN import GameLogic
from gamegrphicN import GameGrphic

from modelN import Linear_QNet, QTrainer
from helperN import plot, save_plot, plot_bar_scores

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001
MAX_RETRAIN_ITERATIONS = 20
class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = 1   # random
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        input_size = 9  # Define your input size
        hidden_size = 128 # Define your hidden layer size
        output_size = 4  # Define your output size
        self.model = Linear_QNet(input_size, hidden_size, output_size) 
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)
        # TODO: model, trainer

    '''
    def get_state(self, game:AntgameAIO.AntgameAIO):


        DIRECTIONS = {
                'up': (-1, 0),
                'down': (1, 0),
                'left': (0, -1),
                'right': (0, 1)
            }
        
        # best_dir = game.choose_direction_lookahead()
        best_dir = game.direction
        dir_u = best_dir == list(DIRECTIONS.keys())[0]
        dir_d = best_dir == list(DIRECTIONS.keys())[1]
        dir_l = best_dir == list(DIRECTIONS.keys())[2]
        dir_r = best_dir == list(DIRECTIONS.keys())[3]
        neighbor = game.get_neighborhood()

        state = np.array([
            *neighbor,
            # dir_u, dir_d, dir_l, dir_r,
            ], dtype=int)
        state_str = ', '.join(map(str, state))
        with open('testTRE.txt', 'a') as file:
            file.write(state_str + "\n")
        return state
   
    '''
    
    def get_state(self, game: GameLogic):
        x, y = game.ant_position
        cols = game.N
        state = x * cols + y
        q_table = np.zeros((cols*cols))
        q_table[state] = 1
        neighbor = game.get_neighborhood()
        grid = game.visited.copy().flatten().tolist()
        # Define the directions including diagonals
        DIRECTIONS = {
            'up_left': (-1, -1),
            'up': (-1, 0),
            'up_right': (-1, 1),
            'down': (1, 0),
            'left': (0, -1),
            'right': (0, 1),
            'down_left': (1, -1),
            'down_right': (1, 1)
        }

        danger = np.zeros(cols*cols) # if reward < 0
        food = np.zeros(cols*cols) #  if reward == 1
        empty = np.zeros(cols*cols) # if reward = 0
        pos = np.zeros(cols*cols) # if reward = 0

        for idx, reward in enumerate (grid):
            if reward == 0:
                empty[idx] = 1
            elif reward == 1:
                food[idx] = 1
            else:
                danger[idx] = 1 

        pos[state] = 1
        outboard = 1 if game.outboared else 0
        done = 1 if game.game_over() else 0
        direction = game.direction
        direction_encoding = [0, 0, 0, 0]  # up, down, left, right
        if direction == 'up':
            direction_encoding[0] = 1
        elif direction == 'down':
            direction_encoding[1] = 1
        elif direction == 'left':
            direction_encoding[2] = 1
        elif direction == 'right':
            direction_encoding[3] = 1
        
                   
        '''
        
        for dir, (dx, dy) in DIRECTIONS.items():
            new_position = (x + dx, y + dy)
            
            if 0 <= new_position[0] < game.N and 0 <= new_position[1] < game.N:
                reward_new_pos = game.grid[new_position]
                if reward_new_pos < 0:
                    danger[dir] = 1
                elif reward_new_pos > 0:
                    food[dir] = 1
            else:
                danger[dir] = 1  # If out of bounds, consider it as danger
        normalized_position = [x / game.N, y / game.N]
        direction = game.direction
        direction_encoding = [0, 0, 0, 0]  # up, down, left, right
        if direction == 'up':
            direction_encoding[0] = 1
        elif direction == 'down':
            direction_encoding[1] = 1
        elif direction == 'left':
            direction_encoding[2] = 1
        elif direction == 'right':
            direction_encoding[3] = 1
    
        '''

        # Create the state array
        state = np.array([
            # *normalized_position,
            # danger['up'], danger['down'], danger['left'], danger['right'],
            # danger['up_left'], danger['up_right'], danger['down_left'], danger['down_right'],
            # food['up'], food['down'], food['left'], food['right'],
            # food['up_left'], food['up_right'], food['down_left'], food['down_right'],
            # *neighbor,
            # *grid,
            # *normalized_position,
            # x,y,
            # *neighbor,
            # *direction_encoding,
            # *empty,
            *danger,
            *food,
            *pos,
            outboard


            # game.reward,
            # game.score
            # game.moves_left

        ], dtype=int)
        print(game.visited)
        # print(f'danger{danger}, food{food}, pos {x,y} {pos}, out: {outboard}, done {done}')
        return q_table
    '''
        def get_state(self, game: GameLogic):
        x, y = game.ant_temp
        grid = game.visited.copy().flatten().tolist()
        
        normalized_position = [x / game.N, y / game.N]
        direction = game.direction
        direction_encoding = [0, 0, 0, 0]  # up, down, left, right
        
        if direction == 'up':
            direction_encoding[0] = 1
        elif direction == 'down':
            direction_encoding[1] = 1
        elif direction == 'left':
            direction_encoding[2] = 1
        elif direction == 'right':
            direction_encoding[3] = 1

        state = np.array([
            *normalized_position,
            *grid,
            *direction_encoding
        ], dtype=float)  # Changed to float for normalization

        return state

    '''
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEM is reached

        
    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done): #one step
        self.trainer.train_step(state, action, reward, next_state, done)




    def get_action(self, state):
        #random moves: tradeoff exploration / 
        # self.epsilon = 200 - self.n_games
        self.epsilon = max(0.1, self.epsilon * 0.95)  # Decay epsilon

        final_move = [0, 0, 0, 0] # up, down, left, right
        if random.random() < self.epsilon :
        
            move = random.randint(0, 3)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype = torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1
        
        # print(f'finalmove {final_move}')
            
        
        return final_move
        


 


def train(game:GameLogic):
    # pygame.init()

    N = 3
    m = 1
    moves = 6
    plot_scores = []
    plot_mean_scores = [] 
    total_score = 0
    record = 0  # best score
    max_moves_left= 0

    agent = Agent()
    # agent.model.load()
    game = game
    # gamegraph = GameGrphic(game)
    # gamegraph.draw()
    guess = {}
    bad_list = []
    f = []
    mov = []
    game_played = 600
    g = game.grid.copy()

    t= {}
    while agent.n_games < game_played:
        #get old state
        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT :
        #         # running = False
        #         pygame.quit()
        #         quit()

        state_old = agent.get_state(game)
        #temp = state_old[0:16]

        #get move
        final_move = agent.get_action(state_old)
        # old_state = np.array([game.grid.copy().flatten().tolist() + final_move], dtype=int)
        # print(*old_state)
        decoded_move = game.decode_move(final_move)
        # gamegraph.draw_direction(decoded_move)
        game.move(decoded_move)
        # gamegraph.draw()
        # perform move and new state
        state_new = agent.get_state(game)
        #state_new[0:16] = temp
        reward = game.get_reward()
        if reward < 0:
            reward = -1
        
        print(f'Reward: {reward}')
        mov.append(decoded_move)
        done = game.game_over()
        score = game.get_outcome()
        moves_left = game.moves_left
        if reward  > 0 : 
            reward += (moves - game.moves_left) * 0.01 + game.score
        #print('temp=', temp)


        #train short memory for one step

        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # print(f'game: {agent.n_games}, score{score}')
            g_tuple = tuple(map(tuple, g))
            if score <= 0:
                if g_tuple in t:
                    t[g_tuple] += 1
                else:
                    t[g_tuple] = 1
                
            if score <= 0 and agent.n_games > 500 :
                # agent.train_short_memory(state_old, final_move, reward, state_new, done, score)

                # agent.train_low_score_memory()  # Focus on low-score experiences
                # agent.retrain_until_positive(game)  # Retrain until positive scores

                if score in guess:
                    guess[score] += 1
                else:
                    guess[score] = 1
            # print('game',game.grid)
            # train the long memory, plot result
 

            agent.n_games += 1
            agent.train_long_memory()

            # print(f'games {agent.n_games}')
            if score > record or (score == record and moves_left > max_moves_left):

                # agent.memory.popleft()
                record = score
                max_moves_left = moves_left
                agent.model.save()
 
                #TODO record = 0
                # if agent.n_games > 500:
                #     break
            game.get_copy()
            mov.clear()
            print('Game: ', agent.n_games, 'Score: ', score, 'Record: ', record )
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
        
    print()
    plt.close()
    sorted_guess = dict(sorted(guess.items(), key=lambda item: item[0]))
    plot_bar_scores(sorted_guess.keys(), sorted_guess.values())



def test(game: GameLogic):
    agent = Agent()
    agent.model.load()
    terminated = False
    while not terminated:
        final_move = [0, 0, 0, 0]
        state = agent.get_state(game)
        state0 = torch.tensor(state, dtype = torch.float)
        prediction = agent.model(state0)
        move = torch.argmax(prediction).item()
        print('move', move)
        final_move[move] = 1
        decoded_move = game.decode_move(final_move)
        print(f'move {decoded_move}')
        game.move(decoded_move)
        terminated = game.game_over()
    print(f'score{game.score}')

    



    

if __name__ == '__main__':
    game = GameLogic(3,1)
    train(game)
    test(game)
    # load_model_and_test()