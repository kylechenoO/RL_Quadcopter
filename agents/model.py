import random
import numpy as np
from keras import layers, models, optimizers, backend

class Actor(object):
    def __init__(self, state_size, action_size, action_low, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.state_shape = (self.state_size, )
        self.action_shape = (self.action_size, )
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low
        self.model()

    def model(self):
        layer_input = layers.Input(shape = self.state_shape)
        layer_hidden_1_input = layers.Dense(units = 256, activation = 'relu')(layer_input)
        layer_hidden_1_bn = layers.BatchNormalization()(layer_hidden_1_input)
        layer_hidden_1_output = layers.Activation('relu')(layer_hidden_1_bn)
        layer_hidden_2_input = layers.Dense(units = 512, activation = 'relu')(layer_hidden_1_output)
        layer_hidden_2_bn = layers.BatchNormalization()(layer_hidden_2_input)
        layer_hidden_2_output = layers.Activation('relu')(layer_hidden_2_bn)
        layer_output_input = layers.Dense(units = self.action_size, activation = 'sigmoid')(layer_hidden_2_output)
        layer_output_output = layers.Lambda(lambda x: (x * self.action_range) + self.action_low)(layer_output_input)
        self.model = models.Model(inputs = layer_input, outputs = layer_output_output)

        gd = layers.Input(shape = self.action_shape)
        opt = optimizers.Adam(lr = 0.0001)
        updates = opt.get_updates(params = self.model.trainable_weights, loss = backend.mean(-gd * layer_output_output))
        self.train_fn = backend.function(inputs = [ self.model.input, gd, backend.learning_phase() ], outputs = [], updates = updates)

class Critic(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.state_shape = (self.state_size,)
        self.action_shape = (self.action_size,)
        self.model()

    def model(self):
        layer_sta_input = layers.Input(shape = self.state_shape)
        layer_sta_hidden_1_input = layers.Dense(units = 512, activation = 'relu')(layer_sta_input)
        layer_sta_hidden_1_bn = layers.BatchNormalization()(layer_sta_hidden_1_input)
        layer_sta_hidden_1_output = layers.Activation('relu')(layer_sta_hidden_1_bn)
        layer_sta_hidden_2_input = layers.Dense(units = 256, activation = 'relu')(layer_sta_hidden_1_output)
        layer_sta_hidden_2_bn = layers.BatchNormalization()(layer_sta_hidden_2_input)
        layer_sta_hidden_2_output = layers.Activation('relu')(layer_sta_hidden_2_bn)

        layer_act_input = layers.Input(shape = self.action_shape)
        layer_act_hidden_1_input = layers.Dense(units = 512, activation = 'relu')(layer_act_input)
        layer_act_hidden_1_bn = layers.BatchNormalization()(layer_act_hidden_1_input)
        layer_act_hidden_1_output = layers.Activation('relu')(layer_act_hidden_1_bn)
        layer_act_hidden_2_input = layers.Dense(units = 256, activation = 'relu')(layer_act_hidden_1_output)
        layer_act_hidden_2_bn = layers.BatchNormalization()(layer_act_hidden_2_input)
        layer_act_hidden_2_output = layers.Activation('relu')(layer_act_hidden_2_bn)

        layer_input = [ layer_sta_input, layer_act_input ]
        layer_output = layers.Dense(units = 1)(layers.Activation('relu')(layers.Add()([layer_sta_hidden_2_output, layer_act_hidden_2_output])))

        self.model = models.Model(inputs = layer_input, outputs = layer_output)
        self.model.compile(optimizer = optimizers.Adam(lr = 0.001), loss = 'mse')
        self.get_action_gradients = backend.function(inputs = [ *self.model.input, backend.learning_phase() ], outputs = backend.gradients(layer_output, layer_act_input))

class Noise(object):
    def __init__(self, size, mu, theta, sigma):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return(self.state)
