
"""
This script implements a genetic algorithm to find optimal movements for an agent (like an ant) 
in a grid-based game. It simulates multiple generations of potential solutions (genomes), 
evaluates their performance, and evolves them through selection, crossover, and mutation. 
The best solutions are visualized in a plot.
"""
import random 
from gamelogicF import GameLogic
import numpy as np
import matplotlib.pyplot as plt


POPULATION_SIZE = 200
# GENOME_LENGTH = 12
CROSSOVER_RATE = 0.7
MUTATION_RATE = 0.01
GENERATION =  10
TOURNAMENT_SIZE = 10



def random_genome(length):
    """
    Generates a random genome of specified length consisting of movement codes.
    00 : up, 01 : down, left : 10, right: 11
    """
    return random.choices(['00', '01', '10', '11'], k=length)

def decode_move(moves: list):
    """Decodes binary movement codes into human-readable directions."""
    move_map = {'00': 'up', '01': 'down', '10': 'left', '11': 'right'}
    return [move_map[binary_move] for binary_move in moves]


def fitnes(game:GameLogic,  genome:list, return_moves= False):
    """
    Evaluates the fitness of a genome by simulating the game and calculating the score.
    """

    done = False
    moves_played = []
    while not done:
        move = genome[0]
        game.move(move)
        moves_played.append(move)
        done = game.game_over()
        genome.pop(0)
    score = game.get_outcome() + game.moves_left
    game.get_copy()
    if return_moves:
        return score, moves_played
    return  score

def real_score(game:GameLogic,  genome:list):
    """
    Calculates the real score for the given genome without returning moves.
    """

    done = False
    moves_played = []
    while not done:
        move = genome[0]
        game.move(move)
        moves_played.append(move)
        done = game.game_over()
        genome.pop(0)
    score = game.get_outcome()
    game.get_copy()
    return  score

def init_population(population_size, genome_length):
    """Initializes a population with random genomes."""

    return [random_genome(genome_length) for _ in range(population_size)]


def select_parent(game: GameLogic, population):
    """Selects a parent genome using tournament selection."""

    parents = random.choices(population, k= TOURNAMENT_SIZE)
    fitness_values = [fitnes(game, decode_move(p)) for p in parents]
    min_fitness = min(fitness_values)
    if min_fitness <= 0:
        fitness_values = [f + abs(min_fitness) + 1 for f in fitness_values]  # Shift to make all values positive
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]
    
    return parents[np.random.choice(len(parents), p=selection_probs)]


def select_parent1(game: GameLogic, population):
    parents = population
    fitness_values = [fitnes(game, decode_move(p)) for p in parents]
    min_fitness = min(fitness_values)
    if min_fitness <= 0:
        fitness_values = [f + abs(min_fitness) + 1 for f in fitness_values]  # Shift to make all values positive
    total_fitness = sum(fitness_values)
    selection_probs = [f / total_fitness for f in fitness_values]

    return parents[np.random.choice(len(parents), p=selection_probs)]



def crossover(p1, p2):
    """Performs crossover between two parent genomes."""
    if random.random() < CROSSOVER_RATE:
        crossover_point = random.randint(1, len(p1) - 1)
        return p1[:crossover_point] + p2[crossover_point:], p2[:crossover_point] + p1[crossover_point:]
    else:
        return p1, p2
    

def mutate(genome):
    """Applies mutation to a genome."""

    mutation_map = {
        '00': '11',
        '01': '10',
        '10': '01',
        '11': '00'
    }
    for i in range(len(genome)):
        if random.random() < MUTATION_RATE:
            genome[i] = mutation_map[genome[i]]
    return genome


def genetic_algorthm():
    """Main function to run the genetic algorithm for evolving solutions."""

    grid_length = 4
    ant_view = 1
    game = GameLogic(grid_length, ant_view)
    population = init_population(POPULATION_SIZE, game.N*2)
    scores = []
    for generation in range(GENERATION):
        # fiteness_values = [fitnes(game, decode_move(genome)) for genome in population]
        new_population = []
        for _ in range(POPULATION_SIZE // 2 - 2):
            parent1 = select_parent(game, population)
            parent2 = select_parent(game, population)

            # print(f'p1 {parent1}, p2 {parent2}')
            offspring1, offspring2 = crossover(parent1, parent2)
            new_population.extend([mutate(offspring1), mutate(offspring2)])
            # print(population[:5])
            # print('='*10)
            new_population.extend([parent1, parent2])
  
                
        population = new_population
        fiteness_values = [fitnes(game, decode_move(genome)) for genome in population]
        best_fitness = max(fiteness_values)
        idx = fiteness_values.index(best_fitness)
        _, moves_played = fitnes(game, decode_move(population[idx]), True)
        real_fitness = real_score(game,  decode_move(population[idx]))
        scores.append(real_fitness)


        print(f'Generation {generation} Best Fitness = {best_fitness}, moves {moves_played}, length {len(moves_played)}')

    best_index = fiteness_values.index(max(fiteness_values))
    best_solution = population[best_index]


    print(f' {game.visited} {game.ant_position}' )

    print(f'Best Solution {best_solution}')
    print(f'Best Solution {decode_move(best_solution)}')

    print(f'Best Fitness {real_score(game, decode_move(best_solution))}')
    return scores



def plot(scores):

    # Plotting the best_fitness list
    plt.plot(scores, marker='o', linestyle='-', color='b')
    plt.title("Best Fitness Over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness")
    plt.grid()
    plt.xticks(range(5,len(scores)+1 , 5), rotation = 45)  # Set x-ticks to match the number of points
    plt.show()

def main():
    scores = genetic_algorthm()
    plot(scores)


if __name__ == '__main__':
    main()

