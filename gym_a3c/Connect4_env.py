
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import chainer


class Connect4Env(gym.Env):

    def __init__(self):
        # internal state
        self.board_height=6
        self.board_width=7
        self.start_config = np.array([[0. for _ in range(self.board_width)] for _ in range(self.board_height)], dtype=np.float32)  # 0: empty, 1:agent, -1: opponent (epsilon-greedy agent)
        self.state = np.copy(self.start_config)
        self.global_t = 0

        # rewards and punishments
        self.illegal_move_punishment = -100
        self.won_reward = 100
        self.lost_punishment = -100
        self.draw_reward = 50
        self.move_punishment = -2

        # states and actions
        self.num_actions = self.board_width  # one for each field
        self.action_space = spaces.Discrete(self.num_actions)  # linearize action space into a number 0 to (board_width x board_height)-1
        self.observation_space = spaces.Box(-1, 1, (self.board_width, self.board_height), dtype=np.float32)

        # agent NN that suggests state and action values
        self.agent = None

    def is_col_full(self,col):
        if self.state[0][col] !=0: #if the top entry of the column is populated - column is full
            return True
        else:
            return False


    def find_where_counter_goes_in_col(self,col,input_state=[]):
        if len(input_state)==0:
            input_state = self.state
        current_state = [i[col] for i in input_state]
        current_state.reverse()
        for j in current_state:
            if j ==0:
                return len(current_state)-1-current_state.index(j)

    def step(self, action):
        col = action

        # illegal move, col is full
        if self.is_col_full(col):
            my_rew, am_i_done = self.illegal_move_punishment, True
            return np.array(self.state), my_rew, am_i_done, {}
            ## do shit

        # legal move carry it out
        find_row = self.find_where_counter_goes_in_col(col)
        self.state[find_row][col] = 1
        my_rew, am_i_done = self.reward()

        # opponent's turn if game is not won/draw
        if my_rew != self.draw_reward and my_rew != self.won_reward:
            my_rew, am_i_done = self.opponent_move()  # board gets updated in self.perform_opponent_action()

        return np.array(self.state), my_rew, am_i_done, {}


    # we don't use the seeding, but it's an abstract class method, so we get a warning if we don't implement it
    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets checked as an int elsewhere, so we need to keep it below 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        return [seed1, seed2]


    def has_won(self, player, state=None):
        if state is None:
            state = self.state

        # Check horizontally
        for row in range(self.board_height):
            for col in range(self.board_width - 3):
                if all(state[row][col + i] == player for i in range(4)):
                    return True

        # Check vertically
        for col in range(self.board_width):
            for row in range(self.board_height - 3):
                if all(state[row + i][col] == player for i in range(4)):
                    return True

        # Check diagonally (top-left to bottom-right)
        for row in range(self.board_height - 3):
            for col in range(self.board_width - 3):
                if all(state[row + i][col + i] == player for i in range(4)):
                    return True

        # Check diagonally (bottom-left to top-right)
        for row in range(3, self.board_height):
            for col in range(self.board_width - 3):
                if all(state[row - i][col + i] == player for i in range(4)):
                    return True

        return False


    def is_draw(self):
        # Check if any empty cell exists
        for row in range(self.board_height):
            for col in range(self.board_width):
                if self.state[row][col] == 0:
                    return False

        # If no empty cell exists, it's a draw
        return True



    def reward(self):
        # agent won
        if self.has_won(1):
            return self.won_reward, True
        # draw
        if self.is_draw():
            return self.draw_reward, True

        # nothing special
        return self.move_punishment, False

    def opponent_move(self):
        # perform epsilon-greedy actions
        self.perform_opponent_action()

        # opponent won
        if self.has_won(-1):
            return self.lost_punishment, True
        # draw
        if self.is_draw():
            return self.draw_reward, True

        # nothing special, game goes on
        return self.move_punishment, False

    def set_agent(self, agent):
        if self.agent is None:
            self.agent = agent  # agent used to play against the AI that is being trained (either the agent that is being trained itself or a pretrained agent)

    def set_opponent_agent(self, agent):
        self.agent = agent

    def set_global_t(self, global_t):
        self.global_t = global_t

    def reset(self):
        # create random start state
        self.state = self.get_random_board()
        return np.array(self.state)

    def get_random_board(self):
        # by default, X always starts. Who is X and who is O is decided randomly. To train the agent, we thus need to start with an empty board or with a board with one move

        # Simple case: start with an empty board or with a board with one move
        start_into_a_game = True
        if not start_into_a_game:
            start_config = np.array([[0. for _ in range(self.board_width)] for _ in range(self.board_height)], dtype=np.float32)  # 0: empty, 1:agent, -1: opponent (epsilon-greedy agent)
            if np.random.random() < 0.5:
                col = np.random.randint(0,self.board_width)
                row = self.board_height-1 #place it at the bottom
                start_config[row][col] = -1


        else:
            num_moves = np.random.randint(0, 15)  # carry out up to 10 moves
            start_config = np.array([[0. for _ in range(self.board_width)] for _ in range(self.board_height)], dtype=np.float32)  # 0: empty, 1:agent, -1: opponent (epsilon-greedy agent)
            for tries in range(10): # try 10 times to get a valid board
                current_player = -1
                successful_board = True

                for _ in range(num_moves):
                    col = np.random.randint(0,self.board_width)
                    row = self.find_where_counter_goes_in_col(col,start_config)
                    if row is None: #if the col is already full
                        succesful_board = False
                        break # start new try
                    start_config[row][col] = current_player
                    if current_player == -1:
                        current_player = 1
                    else:
                        current_player = -1

                    #check if game is already won
                    if self.has_won(1,start_config) or self.has_won(-1,start_config):
                        tries +=1
                        start_config = np.array([[0. for _ in range(self.board_width)] for _ in range(self.board_height)], dtype=np.float32)  # 0: empty, 1:agent, -1: opponent (epsilon-greedy agent)
                        successful_board = False
                        break #start new try

                if successful_board:
                    return start_config

        return start_config

    def perform_opponent_action(self):

        # look ahead to see whether there exists a move such that the game is won
        for c in range(self.board_width):
            test_state = np.copy(self.state)
            if self.is_col_full(c) == False:
                row = self.find_where_counter_goes_in_col(c)
                test_state[row][c] == -1
                if self.has_won(-1,test_state):
                    self.state = np.copy(test_state)
                    return



        # if not act epsilon-greedily. Act more greedily over time:
        # Firstly, random choice:
        if np.random.random() < max(0.15, 0.4 / np.log10(10 + 0.01*self.global_t)):
            # perform random action: generate all actions, shuffle, take first legal one
            actions = np.random.permutation(self.num_actions)
            for action in actions:
                # action is linearized to 0 to (rowsxcols)-1
                col = action
                if self.is_col_full(col)==False: # legal move
                    row = self.find_where_counter_goes_in_col(col)
                    self.state[row][col] = -1
                    return


        else:
            # query NN for best actions, try successively best ones, take first legal one

            with chainer.no_backprop_mode():
                #pout, _ = self.agent.shared_model.pi_and_v(np.array([self.state]))  # play against global agent
                if self.agent == None:
                    pout_probs = [0.167]*7
                else:
                    pout, _ = self.agent.model.pi_and_v(np.array([self.state]))  # uncomment to play against local agent
                    pout_probs = pout.all_prob.data[0]  # probability of all actions
                pout_top_action_probs = sorted(pout_probs, reverse=True)  # sort best to worst


            # iterate over all actions, take the best legal one
            for ap in pout_top_action_probs:
                # get index (i.e. the action) of the current probability
                # note that the prob array was sorted, so we have to find the action that corresponds to this probability
                if self.agent == None:
                    action = np.random.randint(self.board_width)
                else:
                    action = np.where(pout_probs == ap)[0][0]

                col = action
                if self.is_col_full(col)==False: # legal move
                    row = self.find_where_counter_goes_in_col(col)
                    self.state[row][col] = -1
                    return


        return
