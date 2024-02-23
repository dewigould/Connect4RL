# A3C Reinforcement Learning for Connect 4
In this repo we provide code to train an AI agent to play Connect 4, and a GUI interface to play against the agent.

# Details
This repo implements the Asynchronous Advantage Actor-Critic (A3C) reinforcement learning model to play Connect 4. The board has 7 columns and 6 rows, players take it in turns to drop counters from the top - the winner is the first person to create an unbroken line of four counters (horizontally, vertically, or diagonally).


# Useage
To train the agent:
```
python train_ai_agent.py 8 --steps 5000000
```
Here 8 is the number of processes, and --steps is an optional parameter to run the training for 5,000,000 training steps.

To use the GUI, one must populate ```input_path_to_pretrained_agent``` in ```GUI.py``` and then run
```
python GUI.py
```

Play is performed by clicking on the top box of the column you want to drop your counter in.

# Set-up
This code is adapted from the tic-tac-toe code provided in https://github.com/ruehlef/Physics-Reports
In order to run it, one must replace ```site-packages/chainerrl/experiments/train_agent_async.py``` with the file ```train_agent_async.py``` provided in this repo.
