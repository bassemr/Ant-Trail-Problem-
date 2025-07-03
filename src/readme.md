# 🐜 Ant Trial Game Project
This project explores solving the Ant Trial Problem using reinforcement learning techniques, including **Q-learning**, **Monte Carlo Tree Search (MCTS)**, and **Genetic Algorithms**.

## 🔧 Setup Note

Before running the game, make sure to update the paths for assets:

> Change the `PATH_IMAGES` and `PATH_AUDIO` in:
- `src/game_in_slide/antgame.py`
- `src/Normal Game/humangameedit.py`

---




> Solving the Ant Trial Problem using Q-Learning, Monte Carlo Tree Search (MCTS), and Genetic Algorithms.



---

## 📁 Project Structure

```bash
project-root/
├── src/
│   ├── Normal Game/
│   │   └── Standard ant game implementation
│   │
│   ├── game_in_slide/
│   │   └── Game described in the slide deck
│   │
│   ├── Neural Network/
│   │   ├── Qlearn/
│   │   │   └── Q-learning to find sugar
│   │   └── Training/
│   │       └── Failed experiment logs and attempts
│   │
│   ├── Monte Carlo/
│   │   └── mcts.py — MCTS-based solver
│   │
│   └── Final/
│       ├── main.py — Runs both agents:
│       │   ├── Q-learning-based neural agent
│       │   └── Tree-based (MCTS) agent
│       ├── genetic.py — Genetic algorithm for optimization
│       └── data.arff — Dataset for Weka analysis


