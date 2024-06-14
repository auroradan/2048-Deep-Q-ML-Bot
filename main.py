import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers
from Game2048Env import Game2048Env
import numpy as np
import random
from collections import deque

def build_model(input_shape, output_shape):
    model = tf.keras.Sequential()
    model.add(layers.Input(shape=input_shape))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(output_shape, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model

def save_model(model, path="dqn_2048_model.h5"):
    model.save(path)
    print(f"Model saved to {path}")

def load_model(path="dqn_2048_model.h5"):
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        print(f"Model loaded from {path}")
        return model
    else:
        print(f"No model found at {path}, training a new model.")
        return None

def train_dqn(env, model, episodes=1000, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, batch_size=64):
    memory = deque(maxlen=2000)
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_moves = 0
        while not done:
            if np.random.rand() <= epsilon:
                action = random.choice([0, 1, 2, 3])
            else:
                q_values = model.predict(state.reshape(1, *state.shape))
                action = np.argmax(q_values[0])
            next_state, reward, done, _ = env.step(action)
            memory.append((state, action, reward, next_state, done))
            state = next_state
            total_moves += 1

            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for s, a, r, s_next, d in minibatch:
                    target = r
                    if not d:
                        target += gamma * np.amax(model.predict(s_next.reshape(1, *s_next.shape))[0])
                    target_f = model.predict(s.reshape(1, *s.shape))
                    target_f[0][a] = target
                    model.fit(s.reshape(1, *s.shape), target_f, epochs=1, verbose=0)

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Episode {episode + 1}/{episodes} - Moves: {total_moves}")

    save_model(model)

model = load_model() or build_model((4, 4), 4)

env = Game2048Env()
train_dqn(env, model, episodes=1000)
