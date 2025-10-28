"""
Ant Trail Game Instructions
Objective:
    You are controlling an ant on a mission to collect sugar scattered across a grid. You need to collect all the sugar before you run out of moves. The number of moves you have is determined by the size of the grid. The game ends when either:

    You collect all the sugar.
    You use up all your available moves.

How to Play:

    Grid Layout:

        The game takes place on a grid (e.g., 5x5). Each cell in the grid can contain either empty space, sugar, or the ant.
        Your Goal:

        The ant must collect all the sugar on the grid before your moves run out.
        Moves:

        You have 2 times the grid's side length worth of moves. For example, in a 5x5 grid, you will have 2 * 5 = 10 moves to complete the game.

    Collecting Sugar:

        Each time the ant moves onto a cell containing sugar, it collects it, and the sugar is removed from the grid.

    Ant Movement:

        The ant can move in four directions:
        Up (W): Moves the ant up by 1 cell.
        Down (S): Moves the ant down by 1 cell.
        Left (A): Moves the ant left by 1 cell.
        Right (D): Moves the ant right by 1 cell.

    Winning the Game:
        If you collect all the sugar before running out of moves, you win the game!

    Losing the Game:
        If you move out of the grid, you lose.
        If you run out of moves before collecting all the sugar, the game ends and you lose.

"""
# IMPORT LIBRARIES
import pygame
import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import os 

# Initialize Pygame
pygame.init()
pygame.font.init()
pygame.mixer.init()

# Constants
W = 600
H = 600 
FPS = 20

"""
HERE YOU SHOULD TO INSERT THE IAMGES AND AUDIO PATH

"""
#IMAGES AND AUDIO PATH
# PATH_IMAGES = 'E:\\ant_game\\images\\'
# PATH_AUDIO = 'E:\\ant_game\\audio\\'
os.chdir(os.path.dirname(os.path.abspath(__file__)))

PATH_IMAGES = '..\\..\\images\\'
PATH_AUDIO = '..\\..\\audio\\'

# IMAGES
ANT_WALKING = pygame.image.load(PATH_IMAGES+'walk.png')
ANT = pygame.image.load(PATH_IMAGES + 'ant_game.png')
ANT_SUGAR = pygame.image.load(PATH_IMAGES + 'ant_sugar1.png')
ANT_THINKING = pygame.image.load(PATH_IMAGES + 'ant_thinking.png')
ANT_HAPPY = pygame.image.load(PATH_IMAGES + 'ant_happy.png')
ANT_HAPPY2 = pygame.image.load(PATH_IMAGES + 'ant_happy2.jpg')
SUGAR = pygame.image.load(PATH_IMAGES + 'sugar.png')
BROKEN = pygame.image.load(PATH_IMAGES + 'street_broken.jpg')
STREET = pygame.image.load(PATH_IMAGES + 'street.jpg')
GAME_OVER = pygame.image.load(PATH_IMAGES + 'ant_sad.png')
ANT_OUT_BOARD = pygame.image.load(PATH_IMAGES + 'ant_dead.jpeg')
#AUDIO
GAME_OVER_LOSER = PATH_AUDIO + 'lose.wav'
GAME_OVER_WINNER = PATH_AUDIO + 'winner.wav'
STEP = PATH_AUDIO + 'walk.wav'
REWARD = PATH_AUDIO + 'reward.wav'
CRY = PATH_AUDIO + 'cry.wav'
CHEAT = PATH_AUDIO + 'cheat.wav'
GAME = PATH_AUDIO + 'game.wav'
START = PATH_AUDIO + 'start.wav'
# Colors
STARTE = (255, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE =(0, 0, 255)

"""

"""
class AntgameAI:
    def __init__(self, N, m):
        """
        Initializes the Ant Trail Game with a specified grid size, ant view.
        Sets up initial positions for the ant, sugar, and other game parameters.
        """
        # pygame.init()
        # pygame.font.init()
        # pygame.mixer.init()

        self.N = N # grid length
        self.m = m # ant view
        self.reset_game()
        self.screen = pygame.display.set_mode((W, H))
        self.display = pygame.display.set_caption('Ant Trail Game') 
        self.clock = pygame.time.Clock()
        self.screen.fill(BLACK)



    def reset_game(self):
        # pygame.init()
        # pygame.font.init()
        """
        Resets all game parameters to their initial states.
        This function is useful when restarting the game or setting up a new game.
        """
        self.directions = ['up', 'down', 'left', 'right']
        self.grid = self.create_grid(self.N) 
        self.ant_position = self.get_random_empty_position()
        # self.grid[self.ant_position] =-1
        self.score = 0
        self.moves_left = 2 * self.N
        self.direction = None
        self.font = pygame.font.Font(None, 30)
        self.data = []
        self.iteration = 0
        self.reward = 0
        self.game_over = False
        self.thinking = False
        self.out_board = False
        self.sugar_found = False
        self.frames= self.walking_ant()
        self.prev_dir = 'up'





    def create_grid(self, N) -> np.ndarray:
        """ 
        Creates an NxN grid with food placed at random positions.
        
        Parameters:
        N (int): The size of the grid (NxN).
        
        Returns:
        grid (numpy.ndarray): A grid where some cells contain food (represented by 1), and other cells are empty (represented by 0).
        """
        grid = np.zeros((N, N), dtype=int)
        
        food_cells = random.sample(range(N * N), N)
        for cell in food_cells:
            grid[cell // N, cell % N] = 1
        return grid

    def get_random_empty_position(self) -> tuple :
        """
        Returns a random empty position (row, column) from the grid where the value is 0.
        """
        empty_positions = np.argwhere(self.grid == 0)
        #ant_position = tuple(empty_positions[random.choice(range(len(empty_positions)))])
        return tuple(empty_positions[random.choice(range(len(empty_positions)))])

    def move_ant(self, direction) -> int:

        """
        Moves the ant in the specified direction, updates its position, and manages rewards and penalties.
        If the ant moves out of the grid, applies a penalty and ends the game.
        Returns the reward obtained from the move.
        """
        x, y = self.ant_position
        reward = 0
        if direction == 'up':
            new_position = (x-1 , y)
        elif direction == 'down':
            new_position = (x+1 , y)
        elif direction == 'left':
            new_position = (x, y-1 )
        elif direction == 'right':
            new_position = (x, y+1 )
        else:
            new_position = self.ant_position  # Stay in place
        if (0 <= new_position[0] < self.N and 0 <= new_position[1] < self.N )  :
            self.ant_position = new_position
            self.score += self.grid[new_position]
            reward = self.grid[new_position]
            self.sugar_found = False
            if reward == 1:
                self.grid[new_position] -= 1
                self.sugar_found = True
            else:
                self.sugar_found = False
            self.grid[x,y] -= 1
            self.moves_left -= 1

        else:
            reward = - (self.N + 2)
            self.score -= (self.N + 2)  # Penalty for moving out of the grid
            self.moves_left = 0  # End game if ant moves out of the grid
            self.out_board = True
        if self.moves_left == 0 or self.score == self.N:
            self.game_over = True
        # print(self.grid)
        return reward
        
    def walking_ant(self) -> dict:
        """
        Loads and prepares the walking animation frames for the ant in all directions.
        
        Returns:
        dict: A dictionary containing lists of frames for each direction ('up', 'down', 'left', 'right').
        """
        sprite_sheet = ANT_WALKING.convert_alpha()
        # Define the dimensions of each frame and number of frames per direction
        sprite_width = 196
        sprite_height = 248
        num_frames = 7  # Number of frames per direction

        # Create lists to store frames for each direction
        cell_size = H // (self.N * 2)

        # Extract frames for each direction
        frames = {'up': [], 'down': [], 'left': [], 'right': []}
        for direction in frames.keys():
            for i in range(num_frames):
                rect = pygame.Rect(0, i * sprite_height, sprite_width, sprite_height)
                portion = sprite_sheet.subsurface(rect)
                if direction == 'down':
                    portion = pygame.transform.flip(portion, False, True)  # Flip vertically
                elif direction == 'left':
                    portion = pygame.transform.rotate(portion, 90)  # Rotate 90 degrees
                elif direction == 'right':
                    portion = pygame.transform.rotate(portion, -90)  # Rotate -90 degrees
                # Scale the portion to fit the cell size
                portion = pygame.transform.scale(portion, (cell_size, cell_size))
                frames[direction].append(portion)
        return frames

    def draw_direction(self, ant_direction):
        """
        Animates the ant's movement in the specified direction across multiple frames.
        
        Parameters:
        ant_direction (str): The direction the ant is moving ('up', 'down', 'left', 'right').
        """
        cell_size = H // self.N
        grid_x, grid_y = self.ant_position  # Get the ant's position in the grid
        x = grid_y * cell_size  # Convert grid coordinates to pixel coordinates
        y = grid_x * cell_size

        sprite_width = 196
        sprite_height = 248
        num_frames = 7  # Number of frames per direction

        dx, dy = 0, 0
        speed = cell_size // num_frames  # Ensure the ant moves one cell over num_frames frames

        if ant_direction == 'right':
            dx = speed
        elif ant_direction == 'left':
            dx = -speed
        elif ant_direction == 'up':
            dy = -speed
        elif ant_direction == 'down':
            dy = speed

        # Animate the movement over `num_frames` frames
        for i in range(num_frames):
            ant_current_frame = i % num_frames
            current_image = self.frames[ant_direction][ant_current_frame]

            # Clear the screen and redraw the necessary elements
            self.screen.fill(BLACK)
            self.draw_grid()
            self.display_score()
            centered_x = x + i * dx + (cell_size - current_image.get_width()) // 2
            centered_y = y + i * dy + (cell_size - current_image.get_height()) // 2

            # Draw the current frame at the updated, centered position
            self.screen.blit(current_image, (centered_x, centered_y))

            # Update the display
            pygame.display.flip()

            # Control the speed of the animation
            pygame.time.Clock().tick(20)

        # Update the grid position after completing the movement


    def draw_grid(self):

        """
        Draws the grid, including the ant and sugar positions, on the screen.
        The appearance of the ant changes based on its state (thinking, out of bounds, found sugar).
        """

        cell_size = H//self.N
        x,y = self.ant_position
        if self.thinking: 
            ant_image = pygame.transform.scale(ANT_THINKING, (cell_size,cell_size))
        elif self.out_board:
            ant_image = pygame.transform.scale(ANT_OUT_BOARD, (cell_size,cell_size))
        
        elif self.sugar_found:
            ant_image = pygame.transform.scale(ANT_SUGAR, (cell_size,cell_size))
        
        else:
            current_image = self.frames[self.prev_dir][-1]
            ant_image = current_image#pygame.transform.scale(current_image, (cell_size,cell_size))

        sugar_image = pygame.transform.scale(SUGAR, (cell_size,cell_size))
        broken_image = pygame.transform.scale(BROKEN, (cell_size,cell_size))
        street_image = pygame.transform.scale(STREET, (cell_size,cell_size))

        for i in range(self.N):
            for j in range(self.N):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                self.screen.blit(street_image, rect)
                if self.grid[i, j] == 1 and (i != x or j != y ) :
                    # pygame.draw.rect(self.screen, GREEN, rect)
                    self.screen.blit(sugar_image, rect)

                elif self.grid[i, j] < 0:      
                    # pygame.draw.rect(self.screen, ORANGE, rect)
                    self.screen.blit(broken_image, rect)

                pygame.draw.rect(self.screen, WHITE, rect, 1)

        # ant_rect = pygame.Rect(self.ant_position[1] * cell_size, self.ant_position[0] * cell_size, cell_size, cell_size)
        ant_rect_x = (self.ant_position[1] * cell_size) + (cell_size - ant_image.get_width()) // 2
        ant_rect_y = (self.ant_position[0] * cell_size) + (cell_size - ant_image.get_height()) // 2
        ant_rect = pygame.Rect(ant_rect_x, ant_rect_y, ant_image.get_width(), ant_image.get_height())
        if not self.direction:  
            self.screen.blit(ant_image, ant_rect)
    # Create a rect for positioning


    def display_score(self):
        """
        Renders the current score and remaining moves on the screen.
        """
        
        score_text = self.font.render(f'Score: {self.score}', True, RED)
        moves_left_text = self.font.render(f'moves left: {self.moves_left}', True, RED)
        self.screen.blit(score_text, (0,   10))
        self.screen.blit(moves_left_text, (H//2 ,   10))

    def draw_solution_path(self, solution):
        """
        Draws the solution path on the grid, displaying step numbers for each move.
        
        Parameters:
        solution (list): A list of directions representing the path taken by the ant.
        """
        cell_size = H // self.N
        step_number = 1
        temp_ant_pos = self.ant_position
        for direction in solution:
            x, y = temp_ant_pos
            if direction == 'up':
                new_position = (x - 1, y)
            elif direction == 'down':
                new_position = (x + 1, y)
            elif direction == 'left':
                new_position = (x, y - 1)
            elif direction == 'right':
                new_position = (x, y + 1)

            num_text = self.font.render(str(step_number), True, BLUE)
            num_rect = num_text.get_rect(center=(
                new_position[1] * cell_size + cell_size // 2,
                new_position[0] * cell_size + cell_size // 2
            ))
            self.screen.blit(num_text, num_rect)
            
            # # Draw footsteps as small ellipses
            # footstep_rect = pygame.Rect(
            #     new_position[1] * cell_size + cell_size // 4, 
            #     new_position[0] * cell_size + cell_size // 4, 
            #     cell_size // 4, 
            #     cell_size // 4
            # )
            # pygame.draw.ellipse(self.screen, color='yellow', rect=footstep_rect)

            # Update the ant position
            temp_ant_pos = new_position
            step_number += 1
            pygame.display.flip()


    def draw_solution_as_footsteps(self, solution):
        cell_size = H // self.N
        temp = self.ant_position
        green_overlay = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
        green_overlay.fill((0, 255, 0, 128))  # Green
        for step_number, direction in enumerate(solution, start=1):
            x, y = temp
            
            if direction == 'up':
                new_position = (x - 1, y)
            elif direction == 'down':
                new_position = (x + 1, y)
            elif direction == 'left':
                new_position = (x, y - 1)
            elif direction == 'right':
                new_position = (x, y + 1)
            overlay_rect = pygame.Rect(new_position[1] * cell_size, new_position[0] * cell_size, cell_size, cell_size)
            self.screen.blit(green_overlay, overlay_rect)
            # Draw the step number on the grid
            text_surface = self.font.render(str(step_number), True, BLUE)
            text_rect = text_surface.get_rect(center=((new_position[1] * cell_size) + cell_size // 2, 
                                                    (new_position[0] * cell_size) + cell_size // 2))
            self.screen.blit(text_surface, text_rect)
            
            # Update the ant's position
            temp = new_position

            # Draw the grid and the updated ant position

            # Draw the solution steps
            self.screen.fill(BLACK)
            self.draw_grid()
            self.display_score()
            self.screen.blit(text_surface, text_rect)
            self.screen.blit(green_overlay, overlay_rect)
            # Refresh the display

            # Optionally, add a delay to see each step
            pygame.display.flip()
            pygame.time.delay(500)  # 500ms delay
            # self.screen.fill(BLACK)
            # self.draw_grid()
            # self.display_score()
            
    def solution(self) -> list:
        """
        Finds the best path for the ant to collect the maximum amount of sugar.
        
        Returns:
        list: The best path as a list of directions.
        """
        N = self.N
        m = self.m
        max_score = -1
        x,y = self.ant_position

        
        grid = self.grid.copy()
        count_ones = np.count_nonzero(grid == 1)
        best_path = []
        directions = [('right', 0, 1), ('left', 0, -1), ('down', 1, 0), ('up', -1, 0)]
        min_moves_left = self.moves_left
        def backtrack(x, y, moves_left, current_score, current_path):
            remaining_ones = np.count_nonzero(grid == 1)
            nonlocal max_score, best_path, min_moves_left
            if (current_score > max_score)  or (current_score == max_score and moves_left > min_moves_left):
                max_score = current_score
                best_path = current_path[:]
                min_moves_left = moves_left

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



    def play_music(self, music_name):
        """
        Stops any currently playing music and plays the specified sound.
        
        Parameters:
        music_name (str): The file path of the music to play.
        """
        pygame.mixer.stop()
        pygame.mixer.Sound(music_name).play()


    def play(self) -> tuple:

        """
        Main game loop that handles events, updates the game state, and renders the game.
        
        Returns:
        tuple: Contains the current reward, game over status, and score.
        """

        self.screen.fill(BLACK)
        self.draw_grid()
        self.display_score()
        pygame.display.flip()
        self.clock.tick(FPS)
        running = True
        show_solution = False
        #while running and not self.game_over :

        # self.play_music(GAME)
        while running:

            for event in pygame.event.get():
                if event.type == pygame.QUIT :
                # running = False
                    pygame.quit()
                    quit()
                    #break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        self.direction = 'left'
                    elif event.key == pygame.K_RIGHT:
                        self.direction = 'right'
                    elif event.key == pygame.K_UP:
                        self.direction = 'up'
                    elif event.key == pygame.K_DOWN:
                        self.direction = 'down'
                    elif event.key == pygame.K_c:  # Handle the 'C' key press
                        solution = self.solution()
                        show_solution = True
                        self.thinking = True
                        

            if show_solution:
                self.play_music(CHEAT)
                self.draw_solution_as_footsteps(solution)
                pygame.display.flip()
                show_solution = False
                self.thinking = False
                pygame.mixer.stop()

                
            if  self.direction:
                self.draw_direction(self.direction)
                self.prev_dir= self.direction
                reward = self.move_ant(self.direction)
                # print(reward)
                if reward == 1:
                    self.play_music(REWARD)
                elif reward < 0:
                    self.play_music(CRY)
                else:
                    self.play_music(STEP)
                self.direction = None
                running= False

            self.screen.fill(BLACK)
            self.draw_grid()
            self.display_score()
            pygame.display.flip()

            self.clock.tick(FPS)
            
            if self.game_over:
                pygame.time.delay(1000)
                # Display the game over screen
                self.screen.fill(BLACK)
                # self.draw_grid()
                self.display_score()
                if self.score >= 0:
                    self.play_music(GAME_OVER_WINNER)
                    if self.score == self.N:
                        game_over_image = pygame.transform.scale(ANT_HAPPY2, (W, H))
                    else:
                        game_over_image = pygame.transform.scale(ANT_HAPPY, (W, H))

                else:
                    self.play_music(GAME_OVER_LOSER)
                    game_over_image = pygame.transform.scale(GAME_OVER, (W, H))
                
                self.screen.blit(game_over_image, (0, 0))
                game_over_text = self.font.render("GAME OVER", True, RED)
                score_text = self.font.render("score: " + str(self.score), True, RED)
                text_rect = game_over_text.get_rect(topleft=(0 , 10))  # Position the text above the image
                score_rect = game_over_text.get_rect(topleft=(W // 2 , 10))  # Position the text above the image
                self.screen.blit(game_over_text, text_rect)
                self.screen.blit(score_text, score_rect)
                pygame.display.flip()

                # Wait for a few seconds before quitting
                pygame.time.delay(3000)  # 3 seconds delay
                running = False
                # pygame.quit()
                # quit()
    
        return self.reward, self.game_over, self.score

        



def start_game() -> int:

    """
    Initializes the game window, displays instructions, and prompts the user for grid size.
    
    Returns:
    int: The size of the grid (N x N) entered by the user.
    """
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 25)
    pygame.mixer.Sound(GAME).play()
    pygame.display.set_caption('Welcome')

    # Display Instructions
    instructions = [
        "Welcome to the Ant Trail Game!",
        "The objective of the game is to collect as much food as possible.",
        "Use the arrow keys to move the ant.",
        "Press c to cheat and see the best path.",
        "To begin, enter the grid size (N x N):"
    ]

    # Load the background image
    background_image = pygame.image.load(PATH_IMAGES + 'ant3.png')  # Replace with your image file
    background_image = pygame.transform.scale(background_image, (W, H))  # Scale it to fit the screen size

    input_box = pygame.Rect(0, H - 60, 200, 40)
    color_inactive = BLUE
    color_active = RED
    color = color_inactive
    active = False
    text = ''
    placeholder = 'Enter grid size'
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                # If the user clicked on the input_box rect.
                if input_box.collidepoint(event.pos):
                    active = not active
                else:
                    active = False
                color = color_active if active else color_inactive
            if event.type == pygame.KEYDOWN:
                if active:
                    if event.key == pygame.K_RETURN:
                        try:
                            N = int(text)
                            done = True
                        except ValueError:
                            text = ''
                    elif event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    elif event.unicode.isdigit():
                        text += event.unicode

        # Blit the background image
        screen.blit(background_image, (0, 0))

        # Render the instructions
        y = H // 2
        for line in instructions:
            instruction_text = font.render(line, True, BLACK )
            screen.blit(instruction_text, (10, y))
            y += 40

        # Render the current text or placeholder text
        if text:
            txt_surface = font.render(text, True, BLACK)
        else:
            txt_surface = font.render(placeholder, True, WHITE)

        # Resize the box if the text is too long.
        width = max(200, txt_surface.get_width() + 10)
        input_box.w = width

        # Blit the text.
        screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
        # Blit the input_box rect.
        pygame.draw.rect(screen, color, input_box, 2)

        pygame.display.flip()
        clock.tick(30)

    return N



def end_game() -> bool:

    """
    Displays the end game screen with options to play again or exit the game.
    
    Returns:
    bool: True if the player chooses to play again, False otherwise.
    """
    # Initialize Pygame

    # Set up the display

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("End Game Screen")

    # Load the background image
    background_image = pygame.image.load(PATH_IMAGES + 'end_game2.jpg')
    background_image = pygame.transform.scale(background_image, (W, H))

    # Set up font
    font = pygame.font.Font(None, 74)

    # Render text
    play_again_text = font.render('Play Again', True, WHITE)
    bye_text = font.render('Bye', True, WHITE)

    # Create a semi-transparent surface for the mask
    mask = pygame.Surface((W // 2, H))
    mask.set_alpha(180)  # Set transparency level (0 = fully transparent, 255 = fully opaque)
    mask.fill((0, 0, 0))  # Fill it with black color

    play_again = False

    # Main loop
    running = True
    while running:
        mouse_x, mouse_y = pygame.mouse.get_pos()

        # Check for events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if mouse_x < W // 2:
                    play_again = True
                    return True



                    # Restart the game or take appropriate action
                else:
                    # Exit the game
                    pygame.quit()
                    return False
                    # sys.exit()

        # Draw the background image
        screen.blit(background_image, (0, 0))

        # Draw the text without any mask
        # screen.blit(play_again_text, (W // 4 - play_again_text.get_width() // 2, H // 2 - play_again_text.get_height() // 2))
        # screen.blit(bye_text, (3 * W // 4 - bye_text.get_width() // 2, H // 2 - bye_text.get_height() // 2))

        # Apply the mask based on mouse position
        if mouse_x < W // 2:
            # Mouse is on the left side
            screen.blit(mask, (0, 0))  # Apply mask over the left half of the screen
            screen.blit(play_again_text, (W // 4 - play_again_text.get_width() // 2, H // 2 - play_again_text.get_height() // 2))
        else:
            # Mouse is on the right side
            screen.blit(mask, (W // 2, 0))  # Apply mask over the right half of the screen
            screen.blit(bye_text, (3 * W // 4 - bye_text.get_width() // 2, H // 2 - bye_text.get_height() // 2))

        # Re-draw the text so it appears above the mask and stands out

        # Update the display
        pygame.display.flip()

# Call the end_game function to start the program


# Call the end_game function to start the program
def print_instruction():

    # Display instructions for the player
    print("Welcome to the Ant Trail Game!")
    print("Collect all the sugar on the grid before you run out of moves.")
    print(f"Grid size: N x N ")
    print(f"Total moves allowed: 2 * N")
    print("Use W (up), A (left), S (down), D (right) to move the ant.")
    print("Be careful! If you move out of the grid, you lose!")
    print("Let's start!\n")

def run(N, m):
    
    game = AntgameAI(N, m)
    
    while True:
        _, _, score =  game.play()

            
        if game.game_over:
            return score


if __name__ == '__main__':
    print_instruction()
    games_num = 0
    scores = {}
    play_again = True
    N = start_game()
    while play_again:
        games_num += 1
        score = run(N,1)
        if score in scores:
            scores[score] += 1
        else:
            scores[score] = 1

        play_again = end_game()


    # Extract labels and values from the dictionary
    labels = list(scores.keys())
    values = list(scores.values())

    # Create the pie chart
    plt.figure(figsize=(10, 8))  # Increase the figure size
    wedges, texts, autotexts = plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)

    # Add a legend with labels and corresponding values
    plt.legend(wedges, [f'{label}: {value}' for label, value in zip(labels, values)],
            title="Score value : how many times", loc="upper left", bbox_to_anchor=(1, 1), fontsize='small', ncol=1)

    # Add a title
    plt.title(f'Score Distribution Across {games_num} Games')

    # Display the plot
    plt.show(block=True)
    # plt.close()



