from keras import layers, models, optimizers
from keras import backend as K

class Actor:
    def __init__(self, state_size, action_size, action_low, action_high):
        
        self.state_size = state_size #size of ea state
        self.action_size = action_size #size of ea action
        self.action_low = action_low #min val of ea action
        self.action_high = action_high#max val of ea action
        self.action_range = self.action_high - self.action_low #range of actions
        self.build_model()

    def build_model(self):
        # states is input layer
        states = layers.Input(shape=(self.state_size,), name='states')

        # hidden layer 1
        hidden = layers.Dense(units=32, activation='relu')(states)
        #hidden layer 2
        hidden = layers.Dense(units=64, activation='relu')(hidden)
        #hidden layer 3
        hidden = layers.Dense(units=32, activation='relu')(hidden)
        #hidden layer 4
        #hidden = layers.Dense(units=64, activation='relu')(hidden)
        
        # Output layer
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(hidden)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # action val = Q val
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        
        # Define optimizer and training function
        optimizer = optimizers.Adam()
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)
        
class Critic:
    def __init__(self, state_size, action_size):
        self.state_size = state_size #state size
        self.action_size = action_size #action size
        self.build_model()
        
        
    def build_model(self):
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')
        #hidden layers for state pathway
        hidden_s = layers.Dense(units=32, activation='relu')(states)
        #hidden layer 2
        hidden_s = layers.Dense(units=64, activation='relu')(hidden_s)
        #hidden layer for action pathway
        hidden_a = layers.Dense(units=32, activation='relu')(actions)
        #hidden layer 2
        hidden_a = layers.Dense(units=64, activation='relu')(hidden_a)
        #add layers
        combo = layers.Add()([hidden_s, hidden_a])
        combo= layers.Activation('relu')(combo)
        
        
        qValues = layers.Dense(units=1, name='q_values')(combo)
        self.model = models.Model(inputs=[states, actions], outputs=qValues)
        
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer, loss='mse')
        
        grad = K.gradients(qValues, actions)
        
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=grad)

