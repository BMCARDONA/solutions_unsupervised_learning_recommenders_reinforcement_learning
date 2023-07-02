# GRADED_CELL
# UNQ_C1

num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    
    ### START CODE HERE ###     
      tf.keras.layers.Dense(units=256, activation="relu"),
      tf.keras.layers.Dense(units=128, activation="relu"),
      tf.keras.layers.Dense(units=num_outputs, activation="linear")
    ### END CODE HERE ###  
    
])

item_NN = tf.keras.models.Sequential([
    
    ### START CODE HERE ###     
      tf.keras.layers.Dense(units=256, activation="relu"),
      tf.keras.layers.Dense(units=128, activation="relu"),
      tf.keras.layers.Dense(units=num_outputs, activation="linear")
    ### END CODE HERE ###  
])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()