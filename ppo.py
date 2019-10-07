import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
import os
"""
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        #V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        print('delta:', delta)
        gae = delta + gamma * tau * masks[step] * gae
        print('gae:', gae)
        print('gae + values[step]: ', gae + values[step])
        returns.insert(0, gae + values[step])
    return np.array(returns)
"""
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

def decay(step, initial_value, final_step):
    decay = 1.0 - (float(step) / final_step)
    if decay < 0.0:
        decay = 0.0
    return decay * initial_value

def update(actorCritic, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, step, clip_param=0.2, grad_clip=40.0):
    loss_tab = []
    entropy_factor=0.01
    value_factor=0.5
    final_clip_param_step = 1000000
    #clip_param = decay(step, clip_param, final_clip_param_step)

    for epoch in range(ppo_epochs):
        for i in range(int(2048 / mini_batch_size)):
            index = i * mini_batch_size
            batch_actions = _pick_batch(mini_batch_size, actions, i)
            batch_log_probs = _pick_batch(mini_batch_size, log_probs, i)
            batch_obs = _pick_batch(mini_batch_size, states, i, shape=[6, 1])
            batch_returns = _pick_batch(mini_batch_size, returns, i)
            batch_advs = _pick_batch(mini_batch_size, advantages, i)

            newState = tf.constant(batch_obs, tf.float32)
            with tf.GradientTape() as tape:
                dist, value = actorCritic.model(newState)
                distCat = tf.distributions.Categorical(probs=dist)
            
                batch_advs = tf.reshape(batch_advs, [-1, 1])
                batch_advs = tf.cast(batch_advs, tf.float32)
                batch_returns = tf.reshape(batch_returns, [-1, 1])

                value_loss = tf.reduce_mean(tf.square(returns - value))
                value_loss *= value_factor
                
                entropy = tf.reduce_mean(distCat.entropy())
                entropy *= entropy_factor
                log_prob = distCat.log_prob(batch_actions)

                ratio = tf.exp(log_prob - batch_log_probs)
                ratio = tf.reshape(ratio, [-1, 1])
                surr1 = ratio * batch_advs

                surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * batch_advs
                surr = tf.minimum(surr1, surr2)
                policy_loss = tf.reduce_mean(surr)

                loss = value_loss - policy_loss - entropy
                loss_tab.append(loss)
                #print('loss:', loss)
            gradient = tape.gradient(loss, actorCritic.model.trainable_variables)
            gradient, _ = tf.clip_by_global_norm(gradient, grad_clip)

            actorCritic.optimizer.apply_gradients(zip(gradient, actorCritic.model.trainable_variables))

            #batch_features = features_t[:, index, :]
            #batch_masks = self._pick_batch(masks, i)
    """
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):

            newState = tf.constant(state, tf.float32)
            with tf.GradientTape() as tape:
                dist, value = actorCritic.model(newState)
                value = np.reshape(value, value.shape[0])
                distCat = tf.distributions.Categorical(dist)

                entropy = tf.reduce_mean(distCat.entropy())
                entropy *= entropy_factor
                new_log_probs = distCat.log_prob(action)

                ratio = tf.exp(new_log_probs - old_log_probs)
                ratio = tf.reshape(ratio, [-1, 1])
                
                print(ratio.shape)
                surr1 = ratio * advantage
                surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - tf.reduce_mean(tf.minimum(surr1, surr2))
                critic_loss = tf.reduce_mean(tf.square((return_ - value)))
                critic_loss *= value_factor

                print('critic_loss:', critic_loss)
                print('actor_loss:', actor_loss)
                print('entropy:', entropy)
                loss = tf.cast(critic_loss, tf.float32) + tf.cast(actor_loss, tf.float32) - entropy
                print('loss:', loss)
                loss_tab.append(loss)
            
            gradient = tape.gradient(loss, actorCritic.model.trainable_variables)

            actorCritic.optimizer.apply_gradients(zip(gradient, actorCritic.model.trainable_variables))
    """
    return loss_tab

class ActorCritic():
    def __init__(self, num_action, lr, filename='./saved/actor.h5'):
        self.filename = filename

        w = tf.orthogonal_initializer(np.sqrt(2.0))
        self.input = tf.keras.layers.Input(shape=(6,1))

        #x = tf.keras.layers.Conv2D(32, 8, 4, activation=tf.nn.relu)(self.input)
        #x = tf.keras.layers.Conv2D(64, 4, 2, activation=tf.nn.relu)(x)
        #x = tf.keras.layers.Conv2D(64, 3, 1, activation=tf.nn.relu)(x)
        x = tf.keras.layers.Flatten()(self.input)
        x = tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer=w)(x)
        x = tf.keras.layers.Dense(64, activation=tf.nn.relu, kernel_initializer=w)(x)


        policy = tf.keras.layers.Dense(num_action, activation=tf.nn.softmax, kernel_initializer=tf.orthogonal_initializer(0.1))(x)

        value = tf.keras.layers.Dense(1, kernel_initializer=tf.orthogonal_initializer(0.1))(x)
        #out = tf.keras.layers.Lambda(lambda x: x * 2)(out)

        self.model = tf.keras.Model(inputs=[self.input], outputs=[policy, value])

        self.model.summary()

        self.optimizer = tf.train.AdamOptimizer(lr, epsilon=1e-5)

    def save(self):
        self.model.save_weights(self.filename)

    def load(self):
        self.model.load_weights(self.filename)
    
    def hard_copy(self, actor_var):
        [self.model.trainable_variables[i].assign(actor_var[i])
                for i in range(len(self.model.trainable_variables))]