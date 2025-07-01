Remember to change the PATH_IMAGES and PATH_AUDIO in:
    - src\game_in_slide\antgame.py
    - src\Normal Game\humangameedit.py

Project Structure:
- Normal Game: Contains a standard ant game.
- game_in_slide: Contains the game described in the slide.
- Neural Network Training: Contains failed attempts to solve the game.
- Monte Carlo: Contains mcts.py, which solves the game.
- Final:
    - main.py: Hosts two agents:
        - Neural network agent.
        - Tree agent.
    - Genetic algorithm called genetic.py.
    - Data in ARFF format for Weka analysis.
