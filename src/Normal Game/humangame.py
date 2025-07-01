import pygame
import numpy as np
import random
from time import sleep
import sys
import matplotlib.pyplot as plt


# Initialize Pygame
pygame.init()
pygame.font.init()
pygame.mixer.init()

# Constants
W = 600
H = 600 
FPS = 20
#IMAGES
ANT = pygame.image.load('images\\ant_game.png')
ANT_SUGAR = pygame.image.load('images\\ant_sugar1.png')
ANT_THINKING = pygame.image.load('images\\ant_thinking.png')
ANT_HAPPY = pygame.image.load('images\\ant_happy.png')
SUGAR = pygame.image.load('images\\sugar.png')
BROKEN = pygame.image.load('images\\street_broken.jpg')
STREET = pygame.image.load('images\\street.jpg')
GAME_OVER = pygame.image.load('images\\ant_sad.png')
ANT_OUT_BOARD = pygame.image.load('images\\ant_dead.jpeg')
#AUDIO
GAME_OVER_LOSER = 'audio\\lose.wav'
GAME_OVER_WINNER = 'audio\\winner.wav'
STEP = 'audio\\walk.wav'
REWARD = 'audio\\reward.wav'
CRY = 'audio\\cry.wav'
CHEAT = 'audio\\cheat.wav'
GAME = 'audio\\game.wav'
START = 'audio\\start.wav'
# Colors
STARTE = (255, 255, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE =(0, 0, 255)


class AntgameAI:
    def __init__(self, N, m):
        pygame.init()
        pygame.font.init()
        pygame.mixer.init()

        self.N = N
        self.m = m
        self.reset_game()
        self.screen = pygame.display.set_mode((W, H))
        self.display = pygame.display.set_caption('Ant Trail Game')
        self.clock = pygame.time.Clock()
        self.screen.fill(BLACK)

    

    def reset_game(self):
        # pygame.init()
        # pygame.font.init()
        self.directions = ['up', 'down', 'left', 'right']
        self.grid = self.create_grid(self.N) 
        self.ant_position = self.get_random_empty_position()
        self.grid[self.ant_position] =-1
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
        print(self.grid)




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

    def move_ant(self, direction):
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
                self.sugar_found = True
            else:
                self.sugar_found = False
            self.grid[new_position] -= 1
            self.moves_left -= 1

        else:
            reward = - (self.N + 2)
            self.score -= (self.N + 2)  # Penalty for moving out of the grid
            self.moves_left = 0  # End game if ant moves out of the grid
            self.out_board = True
        if self.moves_left == 0 or self.score == self.N:
            self.game_over = True
        return reward
        

    def draw_grid(self):
        cell_size = H//self.N
        if self.thinking: 
            ant_image = pygame.transform.scale(ANT_THINKING, (cell_size,cell_size))
        elif self.out_board:
            ant_image = pygame.transform.scale(ANT_OUT_BOARD, (cell_size,cell_size))
        
        elif self.sugar_found:
            ant_image = pygame.transform.scale(ANT_SUGAR, (cell_size,cell_size))
        
        else:
            ant_image = pygame.transform.scale(ANT, (cell_size,cell_size))

        sugar_image = pygame.transform.scale(SUGAR, (cell_size,cell_size))
        broken_image = pygame.transform.scale(BROKEN, (cell_size,cell_size))
        street_image = pygame.transform.scale(STREET, (cell_size,cell_size))

        for i in range(self.N):
            for j in range(self.N):
                rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
                self.screen.blit(street_image, rect)
                if self.grid[i, j] == 1:
                    # pygame.draw.rect(self.screen, GREEN, rect)
                    self.screen.blit(sugar_image, rect)

                elif self.grid[i, j] < 0:      
                    # pygame.draw.rect(self.screen, ORANGE, rect)
                    self.screen.blit(broken_image, rect)

                pygame.draw.rect(self.screen, WHITE, rect, 1)

        ant_rect = pygame.Rect(self.ant_position[1] * cell_size, self.ant_position[0] * cell_size, cell_size, cell_size)
        self.screen.blit(ant_image,ant_rect)
    def display_score(self):
        
        score_text = self.font.render(f'Score: {self.score}', True, RED)
        moves_left_text = self.font.render(f'moves left: {self.moves_left}', True, RED)
        self.screen.blit(score_text, (0,   10))
        self.screen.blit(moves_left_text, (H//2 ,   10))

    def draw_solution_path(self, solution):
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
            
    def solution(self):
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
                original_value = grid[nx][ny]

                grid[nx][ny] -= 1

                # Recur for the next move
                current_path.append(direction)
                backtrack(nx, ny, moves_left - 1, current_score + original_value, current_path)
                current_path.pop()  # Backtrack: remove the last direction

                # Restore the original value if it was changed
                grid[nx][ny] = original_value

        backtrack(x, y, self.moves_left, 0, [])
        return best_path



    def play_music(self, music_name):
        pygame.mixer.stop()
        pygame.mixer.Sound(music_name).play()









    def play(self):

        #TODO
        # pygame.mixer.stop()


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
                reward = self.move_ant(self.direction)
                print(reward)
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

        
        

        

    #sleep(1000)
    
        return self.reward, self.game_over, self.score
        print(f'SCORE { self.score}')
    '''
    with open('test'+str(self.N)+'.txt', 'w') as file:
        for state in self.data:
            file.write(f"{state}\n")
    '''
        



def start_game():
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
    background_image = pygame.image.load('images\\ant3.png')  # Replace with your image file
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



def end_game():
    # Initialize Pygame

    # Set up the display

    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("End Game Screen")

    # Load the background image
    background_image = pygame.image.load('images\\end_game2.jpg')
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

def run(N, m):
    
    game = AntgameAI(N, m)
    
    while True:
        _, _, score =  game.play()

            
        if game.game_over:
            return score


if __name__ == '__main__':
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
    
    