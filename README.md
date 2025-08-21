# Gomoku AI Agent

## Overview
Gomoku (Five in a Row) is a two-player board game, where players take turn to place pieces on the board, and the first player to get 5 pieces in a row (vertically, horizontally, or diagonally) wins the game. If the board fills up without any 5-in-a-row, the game ends in a draw.

This project creates a prompt-based LLM-powered agent that plays the game of Gomoku on a 8x8 board defined by this [Gomoku AI Framework](https://github.com/sitfoxfly/gomoku-ai). A piece placed by Player 1 is respresented by X, while the a placed by Player 2 is represented by O. Players cannot overwrite existing moves and all moves must be valid (invalid move would be considered as a loss).

The AI agent will:
1. Receive the current board state as input
2. Analyze board states and strategize
2. Return the next move (row, column) as a valid coordinate in JSON format.

## Files

```
.
├── agent1                 
│   ├── agent.json          <-  Agent 1 configuration file
│   └── gomoku_agent.py     <-  Agent 1 implementation
├── agent2                  
│   ├── agent.json          <-  Agent 2 configuration file
│   └── gomoku_agent.py     <-  Agent 2 implementation
├── runs                    <-  Match history visualization
├── arena.ipynb             <-  Arena (agent1 v.s. agent2)
└── secrets.json            <-  Define OPENAI_API_KEY and OPENAI_BASE_URL

```

## Contributors
- [ysgoh97](https://github.com/ysgoh97) 
- [szgan001](https://github.com/szgan001) 
- [momo419685](https://github.com/momo419685)