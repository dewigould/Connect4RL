import tkinter as tk
import tkinter.messagebox

import numpy as np

# import trained network
import chainer
from chainerrl.agents import a3c
from chainerrl import links
from chainerrl.optimizers import rmsprop_async
from chainerrl import policies
from chainer import functions as F

# for GUI
board = tk.Tk()
board.title("Connect 4")
board.resizable(width=False, height=False)
frame = tk.Frame(board)
frame.pack()


num_rows = 6
num_cols = 7
# state of the board
state = np.array([[0. for _ in range(num_cols)] for _ in range(num_rows)], dtype=np.float32)  # 0: empty, -1: player, 1: AI
agent_path = "input_path_to_pretrained_agent"



def is_col_full_given_state(col,state):
    if state[0][col] !=0: #if the top entry of the column is populated - column is full
        return True
    else:
        return False



def find_where_counter_goes_in_col_given_state(col,state):
    current_state = [i[col] for i in state]
    current_state.reverse()
    for j in current_state:
        if j ==0:
            return len(current_state)-current_state.index(j)-1

# define the NN for A3C
class A3CFFSoftmax(chainer.ChainList, a3c.A3CModel):
    def __init__(self, ndim_obs, n_actions, hidden_sizes=(50, 50, 50)):
        self.pi = policies.SoftmaxPolicy(
            model=links.MLP(ndim_obs, n_actions, hidden_sizes, nonlinearity=F.tanh))
        self.v = links.MLP(ndim_obs, 1, hidden_sizes=hidden_sizes, nonlinearity=F.tanh)
        super().__init__(self.pi, self.v)

    def pi_and_v(self, state):
        return self.pi(state), self.v(state)


# load trained A3C agent
model = A3CFFSoftmax(ndim_obs=num_rows*num_cols, n_actions=num_cols)
opt = rmsprop_async.RMSpropAsync()
opt.setup(model)
agent = a3c.A3C(model, opt, t_max=5, gamma=0.99)
agent.load(agent_path)


def check_game(action):
    global player_symbol, CPU_symbol, state, fields

    if action >6:
        tkinter.messagebox.showinfo("cannot click here")
        return
    col = action
    row = find_where_counter_goes_in_col_given_state(col,state)


    if fields[action]["text"] == " ":  # user performed legal move
        fields[row*(num_cols)+col]["text"] = player_symbol
        fields[row*(num_cols)+col]["state"] = "disabled"
        state[row][col] = -1



    # Check for winning conditions
    for row in range(num_rows):
        for col in range(num_cols-3):
            if (state[row][col] == -1 and
                state[row][col+1] == -1 and
                state[row][col+2] == -1 and
                state[row][col+3] == -1):
                tkinter.messagebox.showinfo("Winner", "You won the game.")
                board.quit()
                return

    for row in range(num_rows-3):
        for col in range(num_cols):
            if (state[row][col] == -1 and
                state[row+1][col] == -1 and
                state[row+2][col] == -1 and
                state[row+3][col] == -1):
                tkinter.messagebox.showinfo("Winner", "You won the game.")
                board.quit()
                return

    for row in range(num_rows-3):
        for col in range(num_cols-3):
            if (state[row][col] == -1 and
                state[row+1][col+1] == -1 and
                state[row+2][col+2] == -1 and
                state[row+3][col+3] == -1):
                tkinter.messagebox.showinfo("Winner", "You won the game.")
                board.quit()
                return

    for row in range(num_rows-3):
        for col in range(3, num_cols):
            if (state[row][col] == -1 and
                state[row+1][col-1] == -1 and
                state[row+2][col-2] == -1 and
                state[row+3][col-3] == -1):
                tkinter.messagebox.showinfo("Winner", "You won the game.")
                board.quit()
                return

    # Check for draw condition
    if all(cell != 0 for row in state for cell in row):
        tkinter.messagebox.showinfo("Draw", "The game ended in a draw.")
        board.quit()
        return






    # AI moves next
    fields_index = AI_move()
    fields[fields_index]["text"] = CPU_symbol
    fields[fields_index]["state"] = "disabled"

    # Check for winning conditions
    for row in range(num_rows):
        for col in range(num_cols-3):
            if (state[row][col] == 1 and
                state[row][col+1] == 1 and
                state[row][col+2] == 1 and
                state[row][col+3] == 1):
                tkinter.messagebox.showinfo("Loser! You Suck...", "The AI won the game.")
                board.quit()
                return

    for row in range(num_rows-3):
        for col in range(num_cols):
            if (state[row][col] == 1 and
                state[row+1][col] == 1 and
                state[row+2][col] == 1 and
                state[row+3][col] == 1):
                tkinter.messagebox.showinfo("Loser! You Suck...", "The AI won the game.")
                board.quit()
                return

    for row in range(num_rows-3):
        for col in range(num_cols-3):
            if (state[row][col] == 1 and
                state[row+1][col+1] == 1 and
                state[row+2][col+2] == 1 and
                state[row+3][col+3] == 1):
                tkinter.messagebox.showinfo("Loser! You Suck...", "The AI won the game.")
                board.quit()
                return

    for row in range(num_rows-3):
        for col in range(3, num_cols):
            if (state[row][col] == 1 and
                state[row+1][col-1] == 1 and
                state[row+2][col-2] == 1 and
                state[row+3][col-3] == 1):
                tkinter.messagebox.showinfo("Loser! You Suck...", "The AI won the game.")
                board.quit()
                return

    # Check for draw condition
    if all(cell != 0 for row in state for cell in row):
        tkinter.messagebox.showinfo("Draw", "The game ended in a draw.")
        board.quit()
        return


def AI_move():
    global agent, state, fields
    # statevar = agent.batch_states(np.array([state]), np, agent.phi)

    pout, vout = agent.model.pi_and_v(np.array([state]))
    pout_probs = pout.all_prob.data[0]

    pout_top_action_probs = sorted(pout_probs, reverse=True)
    corresponding_actions = []
    actions_for_output = ""
    # find actions corresponding to the best probabilities
    for ap in pout_top_action_probs:
        position = np.where(pout_probs == ap)[0]
        col = position
        row = find_where_counter_goes_in_col_given_state(col,state)
        corresponding_actions.append(np.where(pout_probs == ap)[0])
        actions_for_output += str(col)+ ": " + str(np.round(ap, 3)) + ", "
    # print actions
    my_probs.delete("1.0", tk.END)
    my_probs.insert(tk.END, "Moves I considered: " + actions_for_output[0:-2])
    my_probs.insert(tk.END, "\nValue of the board: " + str(np.round(vout.data[0][0], 3)))

    # iterate over all actions, take the best legal one
    for action in corresponding_actions:
        col = action[0]

        if is_col_full_given_state(col,state) == False: #legal move
            row = find_where_counter_goes_in_col_given_state(col,state)
            state[row][col] = 1
            return row*(num_cols)+col

    return None  # should never happen, one move needs to be legal


# initialize the rows x columns number of fields
fields = [tk.Button(frame, text=" ", bd=4, height=1, width=2, highlightbackground ="white", font="Helvetica 32 bold", command=lambda i=i: check_game(i), name="field"+str(i)) for i in range(num_cols*num_rows)]



#place in grid
for i in fields:
    row, col = fields.index(i) // num_cols, fields.index(i) % num_cols
    i.grid(row=row,column=col)




my_probs = tk.Text(board, height=60, width=150)
my_probs.pack()
my_probs.insert(tk.END, "Moves I considered: -")
my_probs.insert(tk.END, "\nValue of the board: -")
board.update()


# decide who's X and who's O; X starts
if np.random.random() >= 0.5:
    player_symbol = "X"
    CPU_symbol = "O"
    tkinter.messagebox.showinfo("New game", "You have the X symbol and start the game!")
    board.update()
else:
    player_symbol = "O"
    CPU_symbol = "X"
    tkinter.messagebox.showinfo("New game", "You have the O symbol and the AI starts the game!")
    board.update()
    field_index = AI_move()
    fields[field_index]["text"] = CPU_symbol
    fields[field_index]["state"] = "disabled"

# start game
board.mainloop()
