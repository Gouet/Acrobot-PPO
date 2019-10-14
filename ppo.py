import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import os

def compute_returns(rewards, bootstrap_value, terminals, gamma):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    terminals = np.transpose(terminals, [1, 0])
    returns = []
    R = bootstrap_value
    for i in reversed(range(rewards.shape[0])):
        R = rewards[i] + terminals[i] * gamma * R
        returns.append(R)
    returns = reversed(returns)
    # (T, N) -> (N, T)
    returns = np.transpose(list(returns), [1, 0])
    #returns = np.array(returns)
    return returns

def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    # (N, T) -> (T, N)
    rewards = np.transpose(rewards, [1, 0])
    values = np.transpose(values, [1, 0])
    values = np.vstack((values, [bootstrap_values]))
    terminals = np.transpose(terminals, [1, 0])
    # compute delta
    deltas = []
    for i in reversed(range(rewards.shape[0])):
        V = rewards[i] + (terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))
    # compute gae
    A = deltas[-1,:]
    advantages = [A]
    for i in reversed(range(deltas.shape[0] - 1)):
        A = deltas[i] + (terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)
    # (T, N) -> (N, T)
    advantages = np.transpose(list(advantages), [1, 0])
    return advantages

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = len(states)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[0, rand_ids, :], actions[0, rand_ids], log_probs[0, rand_ids], returns[0, rand_ids], advantage[0, rand_ids]

def _pick_batch(mini_batch_size, data, batch_index, flat=True, shape=None):
    start_index = batch_index * mini_batch_size
    batch_data = data[:, start_index:start_index + mini_batch_size]
    if flat:
        if shape is not None:
            return np.reshape(batch_data, [-1] + shape)
        return np.reshape(batch_data, [-1])
    else:
        return batch_data



def update(actorCritic, time_horizon, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, grad_clip=0.5, value_factor=1, entropy_factor=0.0, shape_state=[6, 1]):
    loss_tab = []

    for epoch in range(ppo_epochs):
        for i in range(int(time_horizon / mini_batch_size)):
            index = i * mini_batch_size
            batch_actions = _pick_batch(mini_batch_size, actions, i)
            batch_log_probs = _pick_batch(mini_batch_size, log_probs, i)
            batch_obs = _pick_batch(mini_batch_size, states, i, shape=shape_state)
            batch_returns = _pick_batch(mini_batch_size, returns, i)
            batch_advs = _pick_batch(mini_batch_size, advantages, i)

            newState = tf.constant(batch_obs, tf.float32)
            with tf.GradientTape() as tape:
                dist, value = actorCritic.model(newState)
                distCat = tf.distributions.Categorical(probs=tf.nn.softmax(dist))
            
                batch_advs = tf.reshape(batch_advs, [-1, 1])
                batch_advs = tf.cast(batch_advs, tf.float32)
                batch_returns = tf.reshape(batch_returns, [-1, 1])
                batch_returns = tf.cast(batch_returns, tf.float32)

                value_loss = tf.reduce_mean(tf.square(batch_returns - value))
                value_loss *= value_factor
                
                entropy = tf.reduce_mean(distCat.entropy())
                entropy *= entropy_factor
                log_prob = distCat.log_prob(batch_actions)

                ratio = tf.exp(log_prob - batch_log_probs)
                ratio = tf.reshape(ratio, [-1, 1])
                surr1 = ratio * batch_advs

                surr2 = tf.clip_by_value(ratio, 1.0 - actorCritic.clip_param, 1.0 + actorCritic.clip_param) * batch_advs
                surr = tf.minimum(surr1, surr2)
                policy_loss = tf.reduce_mean(surr)

                loss = value_loss - policy_loss - entropy
                loss_tab.append(loss)
                if loss is None:
                    print('NONE')
                    print('value_loss', value_loss)
                    print('policy_loss', policy_loss)
                    print('entropy', entropy)
            gradient = tape.gradient(loss, actorCritic.model.trainable_variables)
            gradient, _ = tf.clip_by_global_norm(gradient, grad_clip)

            actorCritic.optimizer.apply_gradients(zip(gradient, actorCritic.model.trainable_variables))

    return tf.reduce_mean(loss_tab)

class ActorCritic():
    def __init__(self, num_action, lr, clip_param, final_step, state_shape, filename='./saved/actor.h5'):
        self.filename = filename

        w = tf.orthogonal_initializer(np.sqrt(2.0))
        #self.input = tf.keras.layers.Input(shape=(6,1))
        self.input = tf.keras.layers.Input(shape=state_shape)

        self.lr = tf.Variable(lr)
        self.initial_learning_rate_value = lr
        self.final_learning_rate_step = final_step

        self.clip_param = clip_param
        self.initial_clip_param = clip_param
        self.final_clip_param_step = final_step

        x = tf.keras.layers.Flatten()(self.input)
        x = tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer=w)(x)
        x = tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer=w)(x)

        policy = tf.keras.layers.Dense(num_action, kernel_initializer=tf.orthogonal_initializer(0.1))(x)

        value = tf.keras.layers.Dense(1, kernel_initializer=tf.orthogonal_initializer(0.1))(x)
        #out = tf.keras.layers.Lambda(lambda x: x * 2)(out)

        self.model = tf.keras.Model(inputs=[self.input], outputs=[policy, value])

        self.model.summary()

        self.optimizer = tf.train.AdamOptimizer(self.lr, epsilon=1e-5)

    def save(self):
        self.model.save_weights(self.filename)

    def decay_learning_rate(self, step):
        decay = 1.0 - (float(step) / self.final_learning_rate_step)
        if decay < 0.0:
            decay = 0.0
        self.lr.assign(decay * self.initial_learning_rate_value)

    def decay_clip_param(self, step):
        decay = 1.0 - (float(step) / self.final_clip_param_step)
        if decay < 0.0:
            decay = 0.0
        self.clip_param = (decay * self.initial_clip_param)

    def load(self):
        self.model.load_weights(self.filename)
    
    def hard_copy(self, actor_var):
        [self.model.trainable_variables[i].assign(actor_var[i])
                for i in range(len(self.model.trainable_variables))]