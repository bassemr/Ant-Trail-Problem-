from gamelogicN import GameLogic
import random
import numpy as np
from gamegrphicN import GameGrphic


num_episodes = 10_000
max_steps_per_episode = 500000
learning_rate = 0.01
discount_rate = 0.9
exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.1
exploration_decay_rate = 0.0001
game = GameLogic(6,1)

print(game.visited)
print(f'Position {game.ant_position}')
cols = game.N
q_table = np.zeros((cols*cols, 4))
rewards_all_episodes = []
record = 0
max_moves = game.moves_left
path = []
best_path = []
best_all = []
record_score = 0
record_moves = 0
# Q-learning algorithm
for episode in range(num_episodes):
    # print(f'HERE {game.ant_position}')
    # print(f'HERE {game.visited}')

    # initialize new episode params
    state = game.ant_position[0] * cols + game.ant_position[1]
    rewards_current_episode = 0
    done = False
    states = []
    states.append(state)
    temp = False

    

    while not done: 
        
        # Exploration-exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        action = [0, 0, 0, 0]
        if exploration_rate_threshold  > exploration_rate:
            move_idx = np.argmax(q_table[state, :])
            action[move_idx] = 1
        else:
            move_idx = random.randint(0, 3)
            action[move_idx] = 1

        decoded_move = game.decode_move(action)
        path.append(decoded_move)
        game.move(decoded_move)
        new_state = game.ant_position[0] * cols + game.ant_position[1]
        done = game.game_over()
        score = game.score
        moves_left = game.moves_left
        reward = game.get_reward()
        # if new_state in states:
        #     reward -= 10
        #     temp = True
        #     print('reward')
        
        states.append(new_state)
         


        if done:
            if  reward >= 0 and score > reward and  ( (score > record_score)  or ((score == record_score) and (moves_left > max_moves) and (not path_exists) and (score > 0) and not temp) ):

            # if reward >= 0 and ( (score > record_score)  or ((score == record_score) and (moves_left > record_moves ) and (score > 0))):
                # print(f'R { reward}')
                print(f'BEFORE Reward { reward}, score{score}, moves left {moves_left}')
                reward += (moves_left + score) * 0.1 
                record_moves = moves_left
                record_score = score



        # Update Q-table for Q(s,a)
        q_table[state, move_idx] = q_table[state, move_idx] * (1 - learning_rate) + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))
        
        state = new_state
        rewards_current_episode += reward 



        if done :

            # print(f'episode {episode}, score {score}')
            path_exists = any(stored_path == path for stored_path, _ in best_all)
            # if reward >= 0 and score > reward and  ( (score > record_score)  or ((score == record_score) and (moves_left > record_moves ) and (score > 0))):
            if  reward >= 0 and score > record or (score == record and moves_left > max_moves and not path_exists ):
                print(f'Reward { reward}, score{score}, moves left {moves_left}, path{path}')
                # agent.memory.popleft()
                record = score
                max_moves = moves_left
                best_path = path[:]
                if best_all and best_all[-1][1] < score:
                    best_all.clear()

                best_all.append((path[:], score))
            
            temp = False
            game.get_copy()
            path.clear()
            # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)
    rewards_all_episodes.append(rewards_current_episode)

# game.get_copy()
# print(f'HERE {game.ant_position}')
# print(f'HERE {game.visited}')
# Calculate and print the average reward per thousand episodes
# rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes),num_episodes/1000)
# count = 1000

# print("********Average reward per thousand episodes********\n")
# for r in rewards_per_thousand_episodes:
#     print(count, ": ", str(sum(r/1000)))
#     count += 1000
#     pass
        # Take new action
        # Update Q-table
        # Set new state
        # Add new reward        

    # Exploration rate decay   
#     # Add current episode reward to total rewards list''

# print('HHHHHHHHHHHHHHHHHHHHH')
ACTIONS = ['U','D','L','R'] 
num_states = len(q_table)
edge = np.sqrt(num_states)
for s in range(num_states):
    #  Format q values for printing
    q_values = ''
    for q in q_table[s, :].tolist():
        q_values += "{:+.2f}".format(q)+' '  # Concatenate q values, format to 2 decimals
    q_values=q_values.rstrip()              # Remove space at the end

    # Map the best action to L D R U
    best_action = ACTIONS[np.argmax(q_table[s,:])]

    # Print policy in the format of: state, action, q values
    # The printed layout matches the FrozenLake map.
    print(f'{s:02},{best_action},[{q_values}]', end=' ')         
    if (s+1)%edge==0:
        print() # Print a newline every 4 states

print(f'pos {game.ant_position}')
print(game.visited)
print(best_path)
print(record)
# print(max_moves)
# print(best_all)
# gamegraph = GameGrphic(game)
# gamegraph.draw()
# for path in best_all:
#     for move in path[0]:
#         gamegraph.draw_direction(move)
#         game.move(move)
#         gamegraph.draw()
#     game.get_copy()

#TODO i want to see the game solved with all the pathes possibile!!!!!

gamegraph = GameGrphic(game)
gamegraph.draw()
done = False
while not done:
    action = [0, 0, 0, 0]
    state = game.ant_position[0] * cols + game.ant_position[1]
    action[np.argmax(q_table[state, :])] = 1
    move = game.decode_move(action)
    print(action, move)
    gamegraph.draw_direction(move)
    game.move(move)
    gamegraph.draw()
    done = game.game_over()
score = game.get_outcome()
print(f' Score {score}')

