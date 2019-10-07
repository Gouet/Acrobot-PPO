import gym
import time
import numpy as np
import ppo
import tensorflow as tf
import tensorflow_probability as tfp
tf.enable_eager_execution()
import os
import cv2
import matplotlib.pylab as plt

#Hyper params:
lr               = 0.0003
num_steps        = 1000
mini_batch_size  = 64
ppo_epochs       = 10
threshold_reward = -200
time_horizon = 2048

try:  
    os.mkdir('./saved')
except OSError:  
    print ("Creation of the directory failed")
else:  
    print ("Successfully created the directory")

print(tf.__version__)
env = gym.make('Acrobot-v1')

global ep_ave_max_q_value
ep_ave_max_q_value = 0
global total_reward
total_reward = 0

def _process_obs(obs):
    obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
    return obs[None, :, :, None] / 256 # Shape (84, 84, 1)

def create_tensorboard():
    global_step = tf.train.get_or_create_global_step()

    logdir = "./logs/"
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()
    return global_step, writer

def to_grayscale(im, weights = np.c_[0.2989, 0.5870, 0.1140]):
    """
    Transforms a colour image to a greyscale image by
    taking the mean of the RGB values, weighted
    by the matrix weights
    """
    tile = np.tile(weights, reps=(im.shape[0],im.shape[1],1))
    return np.sum(tile * im, axis=2) / 256.0


def make_epsilon_greedy_policy(model, nA):
    def policy_fn(observation, epsilon):
        A = np.ones(nA, dtype=float) * epsilon / nA
        dist, value = model.predict(observation)
        best_action = np.argmax(dist)
        A[best_action] += (1.0 - epsilon)
        return A, value
    return policy_fn

def plti(im, h=8, **kwargs):
    """
    Helper function to plot an image.
    """
    y = im.shape[0]
    x = im.shape[1]
    w = (y/x) * h
    plt.figure(figsize=(w,h))
    plt.imshow(im, interpolation="none", **kwargs)
    plt.axis('off')
    plt.show()

global global_step
global_step, writer = create_tensorboard()
actorCritic = ppo.ActorCritic(3, lr)
actorCriticOld = ppo.ActorCritic(3, lr)

try:
    actorCriticOld.load()
except Exception as e:
    print('failed to load')

epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay_steps = 50000
epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)
policy = make_epsilon_greedy_policy(actorCriticOld.model, 3)
episode = 0
opti_step = -1
log_probs = []
values    = []
states    = []
actions   = []
rewards   = []
masks     = []

for episode in range(10000):
    global_step.assign_add(1)

    obs = env.reset()
    obs = np.reshape(obs, (1, 6, 1))
    done = False
    j = 0
    ep_ave_max_q_value = 0
    total_reward = 0

    entropy = 0

    epsilon = 1
    while not done and j < num_steps:
        if episode % 10 == 0:
            env.render()

        dist, value = actorCriticOld.model.predict(obs)
        distCat = tf.distributions.Categorical(probs=dist)
        action = distCat.sample(1)[0]
        obs2, reward, done, info = env.step(action)
        obs2 = np.reshape(obs2, (1,6,1))

        #obs2 = _process_obs(obs2)

        total_reward += reward
        log_prob = distCat.log_prob(action)
        entropy += tf.reduce_mean(distCat.entropy())
        log_probs.append(log_prob)
        values.append(value[0])
        rewards.append([reward / 10])
        masks.append([1 - done])
        
        states.append(obs[0,:])
        actions.append(action)

        #stack = stack[:,:,:,1:]
        #stack = np.concatenate([stack, obs2], axis=-1)

        obs = obs2
        j += 1
        opti_step += 1
        if opti_step > 0 and (opti_step / 1) % time_horizon == 0:

            values = np.array(values)
            rewards = np.array(rewards)
            masks = np.array(masks)
            actions = np.array(actions)
            log_probs = np.array(log_probs)
            states = np.array(states)

            values = np.reshape(values, (1, values.shape[0]))
            rewards = np.reshape(rewards, (1, rewards.shape[0]))
            masks = np.reshape(masks, (1, masks.shape[0]))
            actions = np.reshape(actions, (1, actions.shape[0]))
            log_probs = np.reshape(log_probs, (1, log_probs.shape[0]))
            states = np.expand_dims(states, axis=0)

            _, next_value = actorCritic.model.predict(obs)
            returns = ppo.compute_returns(rewards, next_value[0], masks, 0.99)

            advantage = ppo.compute_gae(rewards, values, next_value[0], masks, 0.99, 0.95)

            advantage = (advantage - np.mean(advantage)) / np.std(advantage)

            indices = np.random.permutation(range(time_horizon))
            states = states[:, indices]
            actions = actions[:, indices]
            log_probs = log_probs[:, indices]
            returns = returns[:, indices]
            advantage = advantage[:, indices]
            masks = masks[:, indices]

            loss = ppo.update(actorCritic, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, opti_step)
            actorCriticOld.hard_copy(actorCritic.model.trainable_variables)

            with writer.as_default(), tf.contrib.summary.always_record_summaries():
                tf.contrib.summary.scalar("loss", loss[0] / float(j))

            actorCritic.save()

            log_probs = []
            values    = []
            states    = []
            actions   = []
            rewards   = []
            masks     = []

    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("entropy", entropy / float(j))
        tf.contrib.summary.scalar("reward", total_reward)
    episode += 1
    print('TOTAL REWARD:', total_reward)

env.close()