# autosweep
Training an ML algorithm to play minesweeper.<br/>
Although the minesweeper game is not fully deterministric, we can train a neural network to play minesweeper for us.

# Demo Video
https://www.youtube.com/watch?v=BfCzAksbTfw

## Requirements
Currently the UI is adjusted to work on ubuntu 22.04 only. Make sure the appearance of the minesweeper matches the pictures used in 'templates' folder.

## install dependencies
pip3 install -r requirements.txt

## Playing 
Both small - 8 by 8 - and large - 30 by 16 - can be started. <br/>
To play the large game (30 by 16) use the following command <br/> 
``` 
python3 play_large_with_seven.py  
```
To play the smaller game (8 by 8) use the following command <br/>
```
python3 play_with_seven.py 
```

# Detailed explination
## Training overview
The steps for training the network follow the logic of batch reinforcement learning.
i) Collect data by playing randomly
ii) Pre-Process the data (including reward shaping)
iii) Train a first net with that data
iv) Play with the first net and record the data
repeat steps ii to iv until the network reaches a human like level. <br/>
A human player will solve roughly 85% of all small - 8 by 8 - and 25% of all large - 30 by 16 - games.

## Decision taking and data collection
Processing the whole game board at once would be difficult and lead to issues when trying to play on a different sized board.
Rather, a local approach of 3 by 3, 5 by 5 and 7 by 7 grids around the location where the gui simulator has clicked is used.
The larger grid - 7 b 7 -  works best for obvious reasons: it contains the most data. <br/>
Once the data is collected a neural net is trained to answer the following question: <b> What is my reward if I click at the center of the current grid? </b> <br/>
This is then run over all possible locations on the game board. The gui simulator then clicks on all locations ranked from highest to lowest until the boad game - formally the state - has changed.<br/>
Once the game state has changed, the deterministic flagging algorithm is run before re-running the inference for the next iteration until the game is lost or won.
<br/>
Data collection currently only works with the 8 by 8 game.

## Data pre-processing
Data pre-processing and reward shaping is vital for multiple reasons.
- Without data normalization, the neural net is prone to overfitt thus reducing the outcome quality.
- Reward shaping is needed to deal with the non-deterministic nature of the game. One same grid can lead to cells being cleared or hitting a mine. To avoid the algorithm of becoming to greedy and thus to aggressive, the maximal reward is capped and the negative reward for hitting a mine is amplified. 

## Neural net architecture
The Neural nets used are very small in size allowing for CPU inference. They contain one or two convolution layers, one or two fully connected layers and one sigmoid layer. 
