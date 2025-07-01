import numpy as np
import random
import pandas as pd
import os
class GameLogic:
    def __init__(self, N, m):
        self.N = N
        self.m = m
        self.reset_game()

    def reset_game(self):
        self.directions = ['up', 'down', 'left', 'right']
        self.grid = self.create_grid(self.N)
        self.ant_position = self.get_random_empty_position()
        self.score = 0
        self.moves_left = 2 * self.N
        self.direction = None
        self.data = []
        self.path= []
        self.last_move = None
        self.reward = 0
        self.gameover = False
        self.visited = np.full((self.N, self.N), -np.inf)
        # self.visited[self.ant_position] = 0
        # self.update_visited()
        self.visited = self.grid.copy()
        self.prev_dir = 'up'
        self.grid_copy= self.grid.copy()
        self.ant_position_copy = self.ant_position[:]
        self.outboared = False



    def get_copy(self):

        self.visited = self.grid_copy.copy()
        self.ant_position = self.ant_position_copy[:] 
        self.gameover = False
        self.moves_left = 2 * self.N
        self.reward = 0
        self.prev_dir = 'up'
        self.score = 0
        self.outboared = False
        self.direction = None

    def create_grid(self, N):
        grid = np.zeros((N, N), dtype=int)
        food_cells = random.sample(range(N * N), N)
        for cell in food_cells:
            grid[cell // N, cell % N] = 1
        return grid
    
    def get_random_empty_position(self):
        empty_positions = np.argwhere(self.grid == 0)
        #ant_position = tuple(empty_positions[random.choice(range(len(empty_positions)))])
        return tuple(empty_positions[random.choice(range(len(empty_positions)))])
    
    def decode_move(self, action):
        move = None
        if np.array_equal(action, [1,0,0,0]):
            move = 'up'
        elif np.array_equal(action, [0,1,0,0]):
            move = 'down'
        elif np.array_equal(action, [0,0,1,0]):
            move = 'left'
        elif np.array_equal(action, [0,0,0,1]):
            move = 'right'
        return move



    def move(self, move):
        x, y = self.ant_position

        self.direction = move
        if self.direction == 'up':
            new_position = (x-1 , y)
        elif self.direction == 'down':
            new_position = (x+1 , y)
        elif self.direction == 'left':
            new_position = (x, y-1 )
        elif self.direction == 'right':
            new_position = (x, y+1 )
        else:
            new_position = self.ant_position  # Stay in place
        if (0 <= new_position[0] < self.N and 0 <= new_position[1] < self.N )  :
            self.ant_position = new_position
            self.reward = self.visited[new_position]
            self.visited[x,y] -= 1
            self.moves_left -= 1
            if self.visited[new_position] == 1:
                self.visited[new_position] -= 1


        else:
            self.reward = -(self.N + 2)
            self.outboared = True
              # Penalty for moving out of the grid
            self.moves_left = 0  # End game if ant moves out of the grid
        # self.ant_temp = new_position
        self.score += self.reward
        self.prev_dir = self.direction

        self.update_visited()

    def game_over(self) -> bool :
        if self.moves_left == 0 or self.score == self.N or \
            (np.all(self.visited != 1) and np.all(self.visited != -np.inf)):
            self.gameover = True
        return self.gameover
        
    def get_neighborhood(self):
        N = self.N
        m = self.m
        x, y = self.ant_position
        
        neighborhood = []  #np.full((2*m+1, 2*m+1), -N-2, dtype=int)  # Default value for out-of-bound cells
        for dx in range(-m, m+1):
            for dy in range(-m, m+1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < N:
                    neighborhood.append(self.visited[nx][ny]) #   [m + dx][m + dy] = self.grid[nx][ny]
                else :
                    neighborhood.append(-N-2)
        return neighborhood #.flatten().tolist()  
    
    def get_legal_moves(self):
        N = self.N
        x,y = self.ant_position
        directions = [('right', 0, 1), ('left', 0, -1), ('down', 1, 0), ('up', -1, 0)]
        grid = self.visited.copy()
        possible_moves = []
        for direction, dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < N and 0 <= ny < N:
                # if grid[nx][ny] >= 0:
                possible_moves.append(direction)
        

        return possible_moves
    
    def get_outcome(self):
        return self.score
    def get_reward(self):
        if self.direction is None:
            return 0
        return self.reward
    

    def undiscover_new_cells(self,x,y,temp_visited, cells) :
        N = self.N
        m = self.m
        for nx,ny in cells:
            if 0 <= nx < N and 0 <= ny < N and temp_visited[nx][ny] != -np.inf:
                temp_visited[nx][ny] = -np.inf
        return temp_visited
    def discover_new_cells(self,x,y,temp_visited) :
        N = self.N
                       
        m = self.m
        cell_discovered = []
        for dx in range(-m, m+1):
            for dy in range(-m, m+1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy

                if 0 <= nx < N and 0 <= ny < N and temp_visited[nx][ny] == -np.inf:
                    temp_visited[nx][ny] = self.grid[nx][ny]
                    cell_discovered.append((nx,ny))
        return temp_visited, cell_discovered
    
    def update_visited(self):
        x, y = self.ant_position
        self.visited, _ = self.discover_new_cells(x,y,self.visited.copy())
    def solution(self):
        N = self.N
        m = self.m
        max_score = -1
        x,y = self.ant_position

        
        grid = self.grid.copy()
        count_ones = np.count_nonzero(grid == 1)
        best_path = []
        besty = []
        directions = [('right', 0, 1), ('left', 0, -1), ('down', 1, 0), ('up', -1, 0)]
        min_moves_left = self.moves_left
        def backtrack(x, y, moves_left, current_score, current_path):
            remaining_ones = np.count_nonzero(grid == 1)
            nonlocal max_score, best_path, min_moves_left
            if (current_score > max_score)  or (current_score == max_score and moves_left > min_moves_left):
                max_score = current_score
                best_path = current_path[:]
                min_moves_left = moves_left
                # if len(besty) > 0:
                #     if besty[-1] [1] < max_score:
                #         besty.clear()
                #         besty.append((best_path, max_score))
                #     else:
                #         besty.append((best_path, max_score))
                # else:
                #     besty.append((best_path, max_score))
                

            if moves_left == 0 or current_score == count_ones :
                return

            # Directions: right, left, down, up
            next_moves = []

            for direction, dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < N and 0 <= ny < N:
                    if grid[nx][ny] >= 0:
                        next_moves.append((direction, nx, ny))

            # Sort moves by grid value, preferring higher values
            next_moves.sort(key=lambda move: grid[move[1]][move[2]] , reverse=True)
            for direction, nx, ny in next_moves:
                original_value = grid[x][y]
                reward = grid[nx][ny]
                if reward == 1:
                    grid[nx][ny] -= 1

                grid[x][y] -= 1

                # Recur for the next move
                current_path.append(direction)
                backtrack(nx, ny, moves_left - 1, current_score + reward, current_path)
                current_path.pop()  # Backtrack: remove the last direction

                # Restore the original value if it was changed
                grid[x][y] = original_value
                if reward == 1:
                    grid[nx][ny] += 1

        backtrack(x, y, self.moves_left, 0, [])
        return best_path
   




