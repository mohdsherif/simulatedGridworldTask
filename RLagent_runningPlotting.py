import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.widgets import Slider
import matplotlib.collections as mcoll
import matplotlib.cm as cm

plt.ion()

# Define the state and action size
# state_shape = (4, 4)
# action_size = 4  # Up, Down, Left, Right


# # Set parameters
# trial_count = 5000
# max_steps = 100
# alpha = 0.5
# gamma = 0.95
# epsilon = 0.1

# SARSA epsilon greedy
def run_SARSA_gridworld_epsilon_greedy(state_shape, action_size, start_state, terminal_state, trial_count, max_steps, alpha, gamma, epsilon, reward_value, randomInitQTable=1234, choice_random_seed=2314, wrap_around_grid=True):
    '''
    Example to run: run_SARSA_gridworld_epsilon_greedy(state_shape=(4, 4), action_size=4, start_state=(0, 0), terminal_state=(3, 3), trial_count=5000, max_steps=100, alpha=0.5, gamma=0.95, epsilon=0.1, reward_value=1, randomInitQTable=1234, choice_random_seed=2314, wrap_around_grid=True)
Will run SARSA for a RL agent in a 2D gridworld composed of state_shape, starting from start_state, with reward at terminal_state, of reward value reward_value, returning paths, list of numpy arrays conveying the path taking by the agent'''
    if randomInitQTable == 'init_with_zero':
        Q_table = np.zeros(state_shape + (action_size,))
    elif type(randomInitQTable) == int: 
        np.random.seed(randomInitQTable)
        Q_table = np.random.rand(state_shape[0], state_shape[1], action_size)
    else:
        print('randomInitQTable must be either "init_with_zero" or an int')
    # Define the reward structure
    rewards = np.zeros(state_shape)
    rewards[terminal_state[0], terminal_state[1]] = reward_value  # Goal
    # TODO: convert to Q_array with dimension for paths 
    # seeding for choose_action
    np.random.seed(choice_random_seed)
    # Define the action function
    def choose_action(state):
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(action_size)  # Explore
        else:
            action = np.argmax(Q_table[state])  # Exploit
        return action
    # Define the state transition function
    def transition(state, action):
        if wrap_around_grid: # wrap around grid
            if action == 0:  # Down
                next_state = ((state[0]-1) % state_shape[0], state[1])
            elif action == 1:  # Up
                next_state = ((state[0]+1) % state_shape[0], state[1])
            elif action == 2:  # Left
                next_state = (state[0], (state[1]-1) % state_shape[1])
            elif action == 3:  # Right
                next_state = (state[0], (state[1]+1) % state_shape[1])
        else: # not wrap around
            if action == 0:  # Down
                next_state = (max(state[0]-1, 0), state[1])
            elif action == 1:  # Up
                next_state = (min(state[0]+1, 3), state[1])
            elif action == 2:  # Left
                next_state = (state[0], max(state[1]-1, 0))
            elif action == 3:  # Right
                next_state = (state[0], min(state[1]+1, 3))
        reward = rewards[next_state]
        return next_state, reward
    paths = []
    # start_state = (0, 0)
    # terminal_state = (3, 3)
    for trial in range(trial_count):
        state = start_state
        action = choose_action(state)
        path = [state]  # Store the path for visualization
        for step in range(max_steps):
            next_state, reward = transition(state, action)
            next_action = choose_action(next_state)
            Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * Q_table[next_state][next_action] - Q_table[state][action])
            # print(Q_table[state][action])
            state = next_state
            action = next_action
            path.append(state)  # Add the new state to the path
            if state == terminal_state:
                break
        paths.append(path)
    paths = [np.array(path) for path in paths] # convert to a list of numpy arrays
    return Q_table, paths

# Q_table, paths = run_SARSA_gridworld(state_shape=(4, 4), start_state=(0, 0), terminal_state=(3, 3), trial_count=5000, max_steps=100, alpha=0.5, gamma=0.95, epsilon=0.1, reward_value=1)


# SARSA beta softmax
def run_SARSA_gridworld_beta_softmax(state_shape, action_size, start_state, terminal_state, trial_count, max_steps, alpha, gamma, beta, reward_value, randomInitQTable=1234, choice_random_seed=2314, wrap_around_grid=True):
    '''
    Example to run: run_SARSA_gridworld_beta_softmax(state_shape=(4, 4), action_size=4, start_state=(0, 0), terminal_state=(3, 3), trial_count=5000, max_steps=100, alpha=0.5, gamma=0.95, beta=0.8, reward_value=1, randomInitQTable=1234, choice_random_seed=2314, wrap_around_grid=True)
Will run SARSA for a RL agent in a 2D gridworld composed of state_shape, starting from start_state, with reward at terminal_state, of reward value reward_value, returning paths, list of numpy arrays conveying the path taking by the agent'''
    if randomInitQTable == 'init_with_zero':
        Q_table = np.zeros(state_shape + (action_size,))
    elif type(randomInitQTable) == int: 
        np.random.seed(randomInitQTable)
        Q_table = np.random.rand(state_shape[0], state_shape[1], action_size)
    else:
        print('randomInitQTable must be either "init_with_zero" or an int')
    # Define the reward structure
    rewards = np.zeros(state_shape)
    rewards[terminal_state[0], terminal_state[1]] = reward_value  # Goal
    # TODO: convert to Q_array with dimension for paths 
    # seeding for choose_action
    np.random.seed(choice_random_seed)
    # Define the action function
    # def choose_action(state):
    #     if np.random.uniform(0, 1) < epsilon:
    #         action = np.random.choice(action_size)  # Explore
    #     else:
    #         action = np.argmax(Q_table[state])  # Exploit
    #     return action
    def choose_action(state):
        if beta == 0: # deterministic
            action = np.argmax(Q_table[state])
        else: 
            action_probs = np.exp(Q_table[state] / beta)
            action_probs /= np.sum(action_probs)
            action = np.random.choice(action_size, p=action_probs)
        return action

    # Define the state transition function
    def transition(state, action):
        if wrap_around_grid: # wrap around grid
            if action == 0:  # Down
                next_state = ((state[0]-1) % state_shape[0], state[1])
            elif action == 1:  # Up
                next_state = ((state[0]+1) % state_shape[0], state[1])
            elif action == 2:  # Left
                next_state = (state[0], (state[1]-1) % state_shape[1])
            elif action == 3:  # Right
                next_state = (state[0], (state[1]+1) % state_shape[1])
        else: # not wrap around
            if action == 0:  # Down
                next_state = (max(state[0]-1, 0), state[1])
            elif action == 1:  # Up
                next_state = (min(state[0]+1, state_shape[0]-1), state[1])
            elif action == 2:  # Left
                next_state = (state[0], max(state[1]-1, 0))
            elif action == 3:  # Right
                next_state = (state[0], min(state[1]+1, state_shape[1]-1))
        reward = rewards[next_state]
        return next_state, reward
    paths = []
    # start_state = (0, 0)
    # terminal_state = (3, 3)
    for trial in range(trial_count):
        state = start_state
        action = choose_action(state)
        path = [state]  # Store the path for visualization
        for step in range(max_steps):
            next_state, reward = transition(state, action)
            next_action = choose_action(next_state)
            Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * Q_table[next_state][next_action] - Q_table[state][action])
            # if Q-learning, the following would be the udpate code:
            # Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * Q_table[next_state][next_action] - Q_table[state][action])

            # print(Q_table[state][action])
            state = next_state
            action = next_action
            path.append(state)  # Add the new state to the path
            if state == terminal_state:
                break
        paths.append(path)
    paths = [np.array(path) for path in paths] # convert to a list of numpy arrays
    return Q_table, paths


def run_algorithm_gridworld_beta_softmax(state_shape, action_size, start_state, terminal_state, trial_count, max_steps, alpha, gamma, beta, reward_value, algorithm='SARSA', randomInitQTable=1234, choice_random_seed=2314, wrap_around_grid=True):
    '''
    Example to run: run_algorithm_gridworld_beta_softmax(state_shape=(4, 4), action_size=4, start_state=(0, 0), terminal_state=(3, 3), trial_count=5000, max_steps=100, alpha=0.5, gamma=0.95, beta=0.8, reward_value=1, algorithm='SARSA', randomInitQTable=1234, choice_random_seed=2314, wrap_around_grid=True)
Will run SARSA for a RL agent in a 2D gridworld composed of state_shape, starting from start_state, with reward at terminal_state, of reward value reward_value, returning paths, list of numpy arrays conveying the path taking by the agent'''
    if randomInitQTable == 'init_with_zero':
        Q_table = np.zeros(state_shape + (action_size,))
    elif type(randomInitQTable) == int: 
        np.random.seed(randomInitQTable)
        Q_table = np.random.rand(state_shape[0], state_shape[1], action_size)
    else:
        print('randomInitQTable must be either "init_with_zero" or an int')
    # Define the reward structure
    rewards = np.zeros(state_shape)
    rewards[terminal_state[0], terminal_state[1]] = reward_value  # Goal
    # TODO: convert to Q_array with dimension for paths 
    # seeding for choose_action
    np.random.seed(choice_random_seed)
    def choose_action(state):
        if beta == 0: # deterministic
            action = np.argmax(Q_table[state])
        else: 
            action_probs = np.exp(Q_table[state] / beta)
            action_probs /= np.sum(action_probs)
            action = np.random.choice(action_size, p=action_probs)
        return action

    # Define the state transition function
    def transition(state, action):
        if wrap_around_grid: # wrap around grid
            if action == 0:  # Down
                next_state = ((state[0]-1) % state_shape[0], state[1])
            elif action == 1:  # Up
                next_state = ((state[0]+1) % state_shape[0], state[1])
            elif action == 2:  # Left
                next_state = (state[0], (state[1]-1) % state_shape[1])
            elif action == 3:  # Right
                next_state = (state[0], (state[1]+1) % state_shape[1])
        else: # not wrap around
            if action == 0:  # Down
                next_state = (max(state[0]-1, 0), state[1])
            elif action == 1:  # Up
                next_state = (min(state[0]+1, state_shape[0]-1), state[1])
            elif action == 2:  # Left
                next_state = (state[0], max(state[1]-1, 0))
            elif action == 3:  # Right
                next_state = (state[0], min(state[1]+1, state_shape[1]-1))
        reward = rewards[next_state]
        return next_state, reward
    paths = []
    # start_state = (0, 0)
    # terminal_state = (3, 3)
    for trial in range(trial_count):
        state = start_state
        action = choose_action(state)
        path = [state]  # Store the path for visualization
        for step in range(max_steps):
            next_state, reward = transition(state, action)
            next_action = choose_action(next_state)
            if algorithm == 'SARSA':
                Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * Q_table[next_state][next_action] - Q_table[state][action])
            elif algorithm == 'Q-learning':
                Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state][action])
            else:
                raise ValueError("Invalid algorithm. Choose either 'SARSA' or 'Q-learning'")
            # print(Q_table[state][action])
            state = next_state
            action = next_action
            path.append(state)  # Add the new state to the path
            if state == terminal_state:
                break
        paths.append(path)
    paths = [np.array(path) for path in paths] # convert to a list of numpy arrays
    return Q_table, paths

    
def run_algorithm_gridworld_beta_softmax_different_start_locations(state_shape, action_size, terminal_state, trial_count, max_steps, alpha, gamma, beta, reward_value, algorithm='SARSA', randomInitQTable=1234, choice_random_seed=2314, wrap_around_grid=True):
    '''
    Example to run: run_algorithm_gridworld_beta_softmax(state_shape=(4, 4), action_size=4, start_state=(0, 0), terminal_state=(3, 3), trial_count=5000, max_steps=100, alpha=0.5, gamma=0.95, beta=0.8, reward_value=1, algorithm='SARSA', randomInitQTable=1234, choice_random_seed=2314, wrap_around_grid=True)
Will run SARSA for a RL agent in a 2D gridworld composed of state_shape, starting from start_state, with reward at terminal_state, of reward value reward_value, returning paths, list of numpy arrays conveying the path taking by the agent'''
    if randomInitQTable == 'init_with_zero':
        Q_table = np.zeros(state_shape + (action_size,))
    elif type(randomInitQTable) == int: 
        np.random.seed(randomInitQTable)
        Q_table = np.random.rand(state_shape[0], state_shape[1], action_size)
    else:
        print('randomInitQTable must be either "init_with_zero" or an int')
    # Define the reward structure
    rewards = np.zeros(state_shape)
    rewards[terminal_state[0], terminal_state[1]] = reward_value  # Goal
    # TODO: convert to Q_array with dimension for paths 
    # seeding for choose_action
    np.random.seed(choice_random_seed)
    def choose_action(state):
        if beta == 0: # deterministic
            action = np.argmax(Q_table[state])
        else: 
            action_probs = np.exp(Q_table[state] / beta)
            action_probs /= np.sum(action_probs)
            action = np.random.choice(action_size, p=action_probs)
        return action

    # Define the state transition function
    def transition(state, action):
        if wrap_around_grid: # wrap around grid
            if action == 0:  # Down
                next_state = ((state[0]-1) % state_shape[0], state[1])
            elif action == 1:  # Up
                next_state = ((state[0]+1) % state_shape[0], state[1])
            elif action == 2:  # Left
                next_state = (state[0], (state[1]-1) % state_shape[1])
            elif action == 3:  # Right
                next_state = (state[0], (state[1]+1) % state_shape[1])
        else: # not wrap around
            if action == 0:  # Down
                next_state = (max(state[0]-1, 0), state[1])
            elif action == 1:  # Up
                next_state = (min(state[0]+1, state_shape[0]-1), state[1])
            elif action == 2:  # Left
                next_state = (state[0], max(state[1]-1, 0))
            elif action == 3:  # Right
                next_state = (state[0], min(state[1]+1, state_shape[1]-1))
        reward = rewards[next_state]
        return next_state, reward
    paths = []
    lmanhattan_distance = []
    # generate different start locations
    lstart_state = generate_states(state_shape, trial_count)
    for start_state in lstart_state:
    # for trial in range(trial_count):
        state = start_state
        action = choose_action(state)
        path = [state]  # Store the path for visualization
        for step in range(max_steps):
            next_state, reward = transition(state, action)
            next_action = choose_action(next_state)
            if algorithm == 'SARSA':
                Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * Q_table[next_state][next_action] - Q_table[state][action])
            elif algorithm == 'Q-learning':
                Q_table[state][action] = Q_table[state][action] + alpha * (reward + gamma * np.max(Q_table[next_state]) - Q_table[state][action])
            else:
                raise ValueError("Invalid algorithm. Choose either 'SARSA' or 'Q-learning'")
            # print(Q_table[state][action])
            state = next_state
            action = next_action
            path.append(state)  # Add the new state to the path
            if state == terminal_state:
                break
        paths.append(path)
        lmanhattan_distance.append(manhattan_distance(start_state, terminal_state))
    paths = [np.array(path) for path in paths] # convert to a list of numpy arrays
    return Q_table, paths, lmanhattan_distance
    
    

def manhattan_distance(start, goal):
    """
    Computes the Manhattan distance between two points in a grid.

    Parameters:
    start (tuple): (x_s, y_s) coordinates of the starting point.
    goal (tuple): (x_r, y_r) coordinates of the goal/reward location.

    Returns:
    int: Minimum number of steps required to reach the goal.
    """
    return abs(goal[0] - start[0]) + abs(goal[1] - start[1])



def calc_stepCountPaths(paths):
    arr_path_length = np.empty((len(paths)))
    arr_path_length[:] = np.nan
    for pathindx, path in enumerate(paths):
        arr_path_length[pathindx] = len(path) - 1
    return arr_path_length


def generate_states(state_shape, num_of_states):
    lgenerated_states = []
    for i in range(num_of_states):
        state = (int(np.random.choice(range(state_shape[0]))), int(np.random.choice(range(state_shape[1]))))
        lgenerated_states.append(state)
    return lgenerated_states



def plot_stepCountPaths(arr_path_length, ax):
    ax = plt.subplots(1, 1)
    ax.plot(arr_path_length)
    ax.grid(True)
    ax.set_ylabel('number of steps')
    ax.set_xlabel('trial index')



def plottingPaths_trials_slider(paths, start_state, terminal_state, state_shape, jitter_scale=0.1):
    '''will allow visualizing the different trials wihin paths (list of numpy arrays of paths taken)
'''
    # jitter_scale = 0.1
    paths_xjitter = [np.random.normal(0, jitter_scale, size=len(path)) for path in paths]
    paths_yjitter = [np.random.normal(0, jitter_scale, size=len(path)) for path in paths]

    trial_count = len(paths)

    # Create a LineCollection for each trial, adding the jitter to the coordinates
    line_collections = [mcoll.LineCollection([np.column_stack((paths[i][j:j+2, 0] + paths_xjitter[i][j:j+2], paths[i][j:j+2, 1] + paths_yjitter[i][j:j+2])) for j in range(len(paths[i])-1)], colors=cm.rainbow(np.linspace(0, 1, len(paths[i])-1))) for i in range(len(paths))]

    ## slider visualization
    myfig, ax = plt.subplots(1, 1)
    # Set grid lines at minor ticks
    ax.set_xticks(np.arange(0.5, state_shape[0], 1), minor=True)
    ax.set_yticks(np.arange(0.5, state_shape[1], 1), minor=True)
    ax.grid(which='minor', color='k', linestyle='-', linewidth=2)

    # Set tick labels at major ticks
    ax.set_xticks(np.arange(0, state_shape[0], 1))
    ax.set_yticks(np.arange(0, state_shape[1], 1))

    # Draw the start and end points of the first trial
    # ax.scatter(start_state[0], start_state[1], color='blue', s=100, label='Start', marker='o')  # Start point
    ax.scatter(paths[0][0][0], paths[0][0][1], color='blue', s=100, label='Start', marker='o')  # Start point
    ax.scatter(terminal_state[0], terminal_state[1], color='red', s=100, label='End', marker='s')  # End point

    ax.set_xlim(-0.5, state_shape[0]-0.5)
    ax.set_ylim(-0.5, state_shape[1]-0.5)
    ax.legend()

    # Add the first LineCollection to the plot
    idx0 = 0
    ax.add_collection(line_collections[idx0])

    axidx = plt.axes([0.25, 0.03, 0.65, 0.03])
    slidx = Slider(axidx, 'trial index', 0, trial_count-1, valinit=idx0, valstep=1, valfmt='%d')

    # update function that displays the LineCollection for the selected trial
    def update(val):
        idx = int(slidx.val)
        # Remove previous LineCollection
        for coll in ax.collections:
            coll.remove()  # Removes the previously added LineCollection
        line_collection = mcoll.LineCollection([np.column_stack((paths[idx][j:j+2, 0] + paths_xjitter[idx][j:j+2], paths[idx][j:j+2, 1] + paths_yjitter[idx][j:j+2])) for j in range(len(paths[idx])-1)], colors=cm.rainbow(np.linspace(0, 1, len(paths[idx])-1)))
        ax.add_collection(line_collection)  # Add the new LineCollection
        # ax.scatter(start_state[0], start_state[1], color='blue', s=100, label='Start', marker='o')  # Start point
        ax.scatter(paths[idx][0][0], paths[idx][0][1], color='blue', s=100, label='Start', marker='o')  # Start point that varies for each trial
        ax.scatter(terminal_state[0], terminal_state[1], color='red', s=100, label='End', marker='s')  # End point
        myfig.canvas.draw_idle()

    slidx.on_changed(update)

    # New code: Add keyboard controls
    def on_key_press(event):
        if event.key == 'right':
            slidx.set_val(min(slidx.val+1, trial_count-1))
        elif event.key == 'left':
            slidx.set_val(max(slidx.val-1, 0))

    myfig.canvas.mpl_connect('key_press_event', on_key_press)
    return myfig, ax, slidx # returning slidx object so we the slide stays functional with multiple figures



# if __name__ == "__main__":
#     state_shape = (4, 7)
#     action_size = 4
#     start_state = (0, 0)
#     terminal_state = (3, 3)
#     trial_count = 50
#     max_steps = 100
#     alpha = 0.8
#     gamma = 0.95
#     beta = 0.3
#     epsilon = 0.1
#     reward_value = 1
#     jitter_scale = 0.1
#     wrap_around_grid = False

#     Q_table, paths = run_SARSA_gridworld_beta_softmax(state_shape=state_shape, action_size=action_size, start_state=start_state, terminal_state=terminal_state, trial_count=trial_count, max_steps=max_steps, alpha=alpha, gamma=gamma, beta=beta, reward_value=reward_value, wrap_around_grid=wrap_around_grid)

#     myfig, mysp, myslide = plottingPaths_trials_slider(paths, start_state=start_state, terminal_state=terminal_state, state_shape=state_shape, jitter_scale=jitter_scale)
