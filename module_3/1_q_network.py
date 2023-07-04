# UNQ_C1
# GRADED CELL

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

# Create the Q-Network
q_network = Sequential([
    ### START CODE HERE ### 
    Input(shape=state_size),
    Dense(units=64, activation="relu"),
    Dense(units=64, activation="relu"),
    Dense(units=num_actions, activation="linear")
    ### END CODE HERE ### 
    ])

# Create the target Q^-Network
target_q_network = Sequential([
    ### START CODE HERE ### 
    Input(shape=state_size),
    Dense(units=64, activation="relu"),
    Dense(units=64, activation="relu"),
    Dense(units=num_actions, activation="linear")
    ### END CODE HERE ###
    ])

### START CODE HERE ### 
optimizer = Adam(learning_rate=ALPHA)
### END CODE HERE ###