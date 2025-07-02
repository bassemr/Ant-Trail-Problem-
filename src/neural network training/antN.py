# Define model
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import torch
import random
import matplotlib.pyplot as plt
import pygame
import numpy as np 
from collections import deque
from gamelogicN import GameLogic
from gamegrphicN import GameGrphic


class DQN(nn.Module):
    def __init__(self, in_states, h1_nodes, out_actions):
        super().__init__()

        # Define network layers
        self.fc1 = nn.Linear(in_states, h1_nodes)   # first fully connected layer
        self.out = nn.Linear(h1_nodes, out_actions) # ouptut layer w

    def forward(self, x):
        x = F.relu(self.fc1(x)) # Apply rectified linear unit (ReLU) activation
        x = self.out(x)         # Calculate output
        return x

# Define memory for Experience Replay
class ReplayMemory():
    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)
    
    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)

# FrozeLake Deep Q-Learning
class FrozenLakeDQL():
    def __init__(self, game:GameLogic):
        self.game = game
        self.H = 512
    # Hyperparameters (adjustable)
    learning_rate_a = 0.01         # learning rate (alpha)
    discount_factor_g = 0.9         # discount rate (gamma)    
    network_sync_rate = 10          # number of steps the agent takes before syncing the policy and target network
    replay_memory_size = 10_000      # size of replay memory
    mini_batch_size = 100          # size of the training data set sampled from the replay memory
    # Neural Network
    loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
    optimizer = None                # NN Optimizer. Initialize later.
    H1 = 512
    ACTIONS = ['U','D','L','R']     # for printing 0,1,2,3 => L(eft),D(own),R(ight),U(p)

    # Train the FrozeLake environment
    def train(self, episodes):
        # Create FrozenLake instance
        env = self.game
        print(env.grid)
        print(env.ant_position)
        h  = 0.1
        num_states = env.N*env.N#16
        num_actions = 4 #4
        
        epsilon = 1 # 1 = 100% random actions
        memory = ReplayMemory(self.replay_memory_size)

        # Create policy and target network. Number of nodes in the hidden layer can be adjusted.
        policy_dqn = DQN(in_states=num_states, h1_nodes=self.H, out_actions=num_actions)
        target_dqn = DQN(in_states=num_states, h1_nodes=self.H, out_actions=num_actions)

        # Make the target and policy networks the same (copy weights/biases from one network to the other)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        print('Policy (random, before training):')
        self.print_dqn(policy_dqn)

        # Policy network optimizer. "Adam" optimizer can be swapped to something else. 
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        # List to keep track of rewards collected per episode. Initialize list to 0's.
        rewards_per_episode = np.zeros(episodes)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count=0
        cols = env.N
        loop_penalty = 2
        record_score = 0
        record_moves = 0
        for i in range(episodes):
            tmp = False
            state = env.ant_position[0] * cols + env.ant_position[1]  # Initialize to state 0
            terminated = False      # True when agent falls in hole or reached goal
            visited_states = set()
            visited_states.add(state)
            states = []
            states.append(state)
            # Agent navigates map until it falls into hole/reaches goal (terminated), or has taken 200 actions (truncated).
            while(not terminated ):
                final_move = [0, 0, 0, 0]
                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = random.randint(0, 3) # actions: 0=left,1=down,2=right,3=up
                else:
                    # select best action            
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

                # Execute action
                final_move[action] = 1
                decoded_move = env.decode_move(final_move)
                # print(f'action {action}, move {final_move} {decoded_move}')
                env.move(decoded_move)
                new_state = env.ant_position[0] * cols + env.ant_position[1]
                reward = env.get_reward()
                score = env.get_outcome()
  
                if new_state in states:
        
                    reward -= 10
                    tmp + True
            
                states.append(new_state)

               # Encourage reaching the goal quickly
                moves_left = env.moves_left
       
                    
                
                terminated = env.game_over()

                if terminated:
                    if reward >= 0 and score > reward and  ( (score > record_score)  or ((score == record_score) and (moves_left > record_moves ) and (score > 0) and not temp)):
                        # if score == env.N :
                        #     reward += (moves_left  ) * 0.1 + score * h
                        #     h += 0.2  
                        # else:
                        reward += (moves_left  + score) * 0.1 + score * h
                        h += 0.1
                        

                        print('XXXXXX'*10)
                        print(f'ٌReward {reward}, moves left: {moves_left}, Score {score}, Reward {reward}')
                        print('XXXXXX'*10)
                        
                        record_score = score
                        record_moves = moves_left


            
                # Save experience into memory
                temp = [[state, action, new_state, reward, terminated, score, moves_left]]
                memory.append((state, action, new_state, reward, terminated, score, moves_left)) 
                # Move to the next state
                state = new_state

                # Increment step counter
                step_count+=1
                self.optimize(temp, policy_dqn, target_dqn)
                
            # Keep track of the rewards collected per episode.
            if score > 0:
                rewards_per_episode[i] = score
            tmp = False

            # Check if enough experience has been collected and if at least 1 reward has been collected
            # if len(memory)>self.mini_batch_size and np.sum(rewards_per_episode)>0:
            if len(memory) > self.mini_batch_size :
                batch_size =  self.mini_batch_size
            else:
                batch_size = len(memory)

            mini_batch = memory.sample(batch_size)
            self.optimize(mini_batch, policy_dqn, target_dqn)        

            # Decay epsilon
            epsilon = max(epsilon - 1/episodes, 0.1)
            # epsilon = max(0.01, 0.995 * epsilon)
            epsilon_history.append(epsilon)

            # Copy policy network to target network after a certain number of steps
            if step_count > self.network_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count=0
            # print(f'episode: {i}, Score: {score}')
            env.get_copy()
        # Close environment
        # env.close()

        # Save policy
        torch.save(policy_dqn.state_dict(), "AntN_dql.pt")

        # Create new graph 
        plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        plt.plot(sum_rewards)
        
        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        plt.plot(epsilon_history)
        
        # Save plots
        plt.savefig('AntN.png')
        self.print_dqn(policy_dqn)

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Get number of input nodes
        num_states = policy_dqn.fc1.in_features

        current_q_list = []
        target_q_list = []
        for state, action, new_state, reward, terminated, score ,_  in mini_batch:

            if terminated: 
                # Agent either reached goal (reward=1) or fell into hole (reward=0)
                # When in a terminated state, target q value should be set to the reward.
                target = torch.FloatTensor([reward])
            else:
                # Calculate target q value 
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + ( self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max())
                        #self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max()
                    )

            # Get the current set of Q values
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)

            # Get the target set of Q values
            target_q = target_dqn(self.state_to_dqn_input(state, num_states)) 
            # Adjust the specific action to the target that was just calculated
            target_q[action] = target
            target_q_list.append(target_q)
                
        # Compute loss for the whole minibatch
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    '''
    Converts an state (int) to a tensor representation.
    For example, the FrozenLake 4x4 map has 4x4=16 states numbered from 0 to 15. 

    Parameters: state=1, num_states=16
    Return: tensor([0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    '''
    def state_to_dqn_input(self, state:int, num_states:int)->torch.Tensor:
        input_tensor = torch.zeros(num_states)
        input_tensor[state] = 1
        return input_tensor

    # Run the FrozeLake environment with the learned policy
    def test(self):
        # Create FrozenLake instancese
        env = self.game
        gamegraph = GameGrphic(env)
        gamegraph.draw()
        print(env.grid)
        print(env.ant_position)
        num_states = env.N*env.N#16
        num_actions = 4 #4
        cols = env.N
        # Load learned policy
        policy_dqn = DQN(in_states=num_states, h1_nodes=self.H, out_actions=num_actions) 
        policy_dqn.load_state_dict(torch.load("AntN_dql.pt"))
        policy_dqn.eval()    # switch model to evaluation mode

        print('Policy (trained):')
        self.print_dqn(policy_dqn)

        state = env.ant_position[0] * cols + env.ant_position[1]  # Initialize to state 0
        # for i in range(episodes):
        terminated = False      # True when agent falls in hole or reached goal
        # Agen navigates map until it falls into a hole (terminated), reaches goal (terminated), or has taken 200 actions (truncated).
        while(not terminated ):  
            final_move = [0, 0, 0, 0]
            # Select best action   
            with torch.no_grad():
                action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()

            # Execute action
            final_move[action] = 1
            decoded_move = env.decode_move(final_move)
            # gamegraph.draw_direction(decoded_move)
            env.move(decoded_move)
            gamegraph.draw()
            reward = env.get_reward()
            print(f'action {action}, move {final_move} {decoded_move}, reward {reward}')
            terminated = env.game_over()
            state= env.ant_position[0] * cols + env.ant_position[1]
        print(f'Score {env.score}')
            # env.get_copy()
        # env.close()

    # Print DQN: state, best action, q values
    def print_dqn(self, dqn):
        # Get number of input nodes
        num_states = dqn.fc1.in_features
        edge = np.sqrt(num_states)

        # Loop each state and print policy to console
        for s in range(num_states):
            #  Format q values for printing
            q_values = ''
            for q in dqn(self.state_to_dqn_input(s, num_states)).tolist():
                q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
            q_values=q_values.rstrip()              # Remove space at the end

            # Map the best action to L D R U
            best_action = self.ACTIONS[dqn(self.state_to_dqn_input(s, num_states)).argmax()]

            # Print policy in the format of: state, action, q values
            # The printed layout matches the FrozenLake map.
            print(f'{s:02},{best_action},[{q_values}]', end=' ')         
            if (s+1)%edge==0:
                print() # Print a newline every 4 states

if __name__ == '__main__':
    game = GameLogic(3,1)
    frozen_lake = FrozenLakeDQL(game)
    frozen_lake.train(100)
    frozen_lake.test()