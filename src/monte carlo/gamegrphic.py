import pygame
import numpy as np
import random

from gamelogic import GameLogic

# # Initialize Pygame
# pygame.init()
# pygame.font.init()
# pygame.mixer.init()

# Constants
W = 600
H = 600 
FPS = 20
#IMAGES
ANT_WALKING = pygame.image.load('images\\walk.png')
ANT = pygame.image.load('images\\ant_game.png')
ANT_SUGAR = pygame.image.load('images\\ant_sugar1.png')
SUGAR = pygame.image.load('images\\sugar.png')
BROKEN = pygame.image.load('images\\street_broken.jpg')
STREET = pygame.image.load('images\\street.jpg')
ANT_OUT_BOARD = pygame.image.load('images\\ant_dead.jpeg')

# Colors
STARTE = (255, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE =(0, 0, 255)

ORANGE =(255, 165, 0)
# SCREEN = pygame.display.set_mode((W, H))
# DISPLAY = pygame.display.set_caption('Ant Trail Game')
# CLOCK = pygame.time.Clock()
# FONT = pygame.font.Font(None, 30)

class GameGrphic:
    def __init__(self, game: GameLogic):
        pygame.init()
        pygame.font.init()
        pygame.mixer.init()
        self.N = game.N
        self.m = game.m
        self.game = game
        self.directions = game.directions
        self.screen = pygame.display.set_mode((W, H))
        self.display = pygame.display.set_caption('Ant Trail Game')
        self.clock = pygame.time.Clock()
        self.font =  pygame.font.Font(None, 30)
        # self.font = FONT
        self.temp = False
        self.frames= self.walking_ant()



       
    def walking_ant(self):
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
        self.temp = True
        cell_size = H // self.N
        grid_x, grid_y = self.game.ant_position  # Get the ant's position in the grid
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
        self.temp = False

        # Update the grid position after completing the movement


    def draw_grid(self):
        cell_size = H//self.N
       
        current_image = self.frames[self.game.prev_dir][-1]
        ant_image = current_image#pygame.transform.scale(current_image, (cell_size,cell_size))

        sugar_image = pygame.transform.scale(SUGAR, (cell_size,cell_size))
        broken_image = pygame.transform.scale(BROKEN, (cell_size,cell_size))
        street_image = pygame.transform.scale(STREET, (cell_size,cell_size))

        for i in range(self.N):
            for j in range(self.N):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                if self.game.visited[i,j] != -np.inf:
                    self.screen.blit(street_image, rect)

                    if self.game.visited[i, j] == 1:
                        # pygame.draw.rect(self.screen, GREEN, rect)
                        self.screen.blit(sugar_image, rect)
                    elif self.game.visited[i, j] < 0:      
                        # pygame.draw.rect(self.screen, BLUE, rect=rect)
                        self.screen.blit(broken_image, rect)

                    pygame.draw.rect(self.screen, WHITE, rect, 1)
        ant_rect_x = (self.game.ant_position[1] * cell_size) + (cell_size - ant_image.get_width()) // 2
        ant_rect_y = (self.game.ant_position[0] * cell_size) + (cell_size - ant_image.get_height()) // 2
        ant_rect = pygame.Rect(ant_rect_x, ant_rect_y, ant_image.get_width(), ant_image.get_height())
        if not self.temp:  
            self.screen.blit(ant_image, ant_rect)



    def display_score(self):
        
        score_text = self.font.render(f'Score: {self.game.score}', True, WHITE)
        moves_left_text = self.font.render(f'moves left: {self.game.moves_left}', True, WHITE)
        self.screen.blit(score_text, (0,   10))
        self.screen.blit(moves_left_text, (400 ,   10))


    def draw(self):

        self.screen.fill(BLACK)
        self.draw_grid()
        self.display_score()
        pygame.display.flip()
        self.clock.tick(FPS)
    



        

