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
import env_wrapper
import rollout

#Hyper params:
TRAIN_MODE = False
NUM_ACTION = 3
ENV_GAME_NAME = 'Acrobot-v1'
VALUE_FACTOR = 1.0
ENTROPY_FACTOR = 0.0
EPSILON = 0.2
LR = 3e-4
LR_DECAY = 'constant'
GRAD_CLIP = 0.5
TIME_HORIZON = 2048
BATCH_SIZE = 64
GAMMA = 0.99
LAM = 0.95
EPOCH = 10
ACTORS = 1
FINAL_STEP = 10e6
print(FINAL_STEP)
STATE_SHAPE = [6, 1]

print(tf.__version__)
try:  
    os.mkdir('./saved')
except OSError:  
    print ("Creation of the directory failed")
else:  
    print ("Successfully created the directory")

def _process_obs(obs):
    return np.reshape(obs, (1, 6, 1))

def _clip_reward(reward):
    return reward / 10.0

def _end_episode(episode, data, max_step, episode_step):
    global_step.assign_add(1)
    print('Episode: ', episode, ' entropy: ', data[0] / float(episode_step), ' reward', data[1], ' global_step: ', max_step, ' episode_step: ', episode_step)
    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar("entropy", data[0] / float(episode_step))
        tf.contrib.summary.scalar("reward", data[1])
        tf.contrib.summary.scalar("episode_step", episode_step)
        

envs = env_wrapper.EnvWrapper(ENV_GAME_NAME, ACTORS, update_obs=_process_obs, update_reward=_clip_reward, end_episode=_end_episode)
rollouts = [rollout.Rollout() for _ in range(ACTORS)]

global ep_ave_max_q_value
ep_ave_max_q_value = 0
global total_reward
total_reward = 0


def create_tensorboard():
    global_step = tf.train.get_or_create_global_step()

    logdir = "./logs/"
    writer = tf.contrib.summary.create_file_writer(logdir)
    writer.set_as_default()
    return global_step, writer

global global_step
global_step, writer = create_tensorboard()
actorCritic = ppo.ActorCritic(NUM_ACTION, LR, EPSILON, FINAL_STEP, STATE_SHAPE)
actorCriticOld = ppo.ActorCritic(NUM_ACTION, LR, EPSILON, FINAL_STEP, STATE_SHAPE)

try:
    actorCriticOld.load()
except Exception as e:
    print('failed to load')

def train(next_value):
    values = []
    rewards = []
    masks = []
    actions = []
    log_probs = []
    states = []

    for rollout in rollouts:
        obs_d, actions_d, rewards_d, values_d, log_probs_d, terminals_d = rollout.get_storage()
        actions.append(actions_d)
        states.append(obs_d)
        rewards.append(rewards_d)
        values.append(values_d)
        log_probs.append(log_probs_d)
        masks.append(terminals_d)

    values = np.array(values)
    rewards = np.array(rewards)
    masks = np.array(masks)
    actions = np.array(actions)
    log_probs = np.array(log_probs)
    states = np.array(states)

    returns = ppo.compute_returns(rewards, next_value, masks, GAMMA)

    advantage = ppo.compute_gae(rewards, values, next_value, masks, GAMMA, LAM)

    advantage = (advantage - np.mean(advantage)) / np.std(advantage)


    indices = np.random.permutation(range(TIME_HORIZON))
    states = states[:, indices]
    actions = actions[:, indices]
    log_probs = log_probs[:, indices]
    returns = returns[:, indices]
    advantage = advantage[:, indices]
    masks = masks[:, indices]

    loss = ppo.update(actorCritic, TIME_HORIZON, EPOCH, BATCH_SIZE, states, actions, log_probs, returns, advantage, GRAD_CLIP, VALUE_FACTOR, ENTROPY_FACTOR, STATE_SHAPE)
    actorCriticOld.hard_copy(actorCritic.model.trainable_variables)

    for rollout in rollouts:
        rollout.flush()

    pass

t = 0
batch_obs = envs.reset()
update_episode = 0

while True:
    if not TRAIN_MODE:
        envs.render(0)
    
    actions_t = []
    dists_t = []
    values_t = []
    dist_cat_t = []
    entropy_t = []

    for stack in batch_obs:
        dist, value = actorCriticOld.model.predict(stack)
        distCat = tf.distributions.Categorical(probs=tf.nn.softmax(dist))
        action = distCat.sample(1)[0]
        entropy_t.append(distCat.entropy())
        actions_t.append(action)
        dists_t.append(dist)
        dist_cat_t.append(distCat)
        values_t.append(value)

    obs2s_t, rewards_t, dones_t = envs.step(actions_t)

    for i in range(ACTORS):
        data = envs.get_variables_at_index(i)
        if len(data) < 2:
            data = [0, 0]
        envs.add_variables_at_index(i, [np.mean(entropy_t[i]) + data[0], rewards_t[i] + data[1]])

    if t > 0 and (t / ACTORS) % TIME_HORIZON == 0 and TRAIN_MODE:
        next_values = np.reshape(values_t, [-1])
        train(next_values)

        if update_episode % 50 == 0:
            actorCritic.save()
        update_episode += 1

    if TRAIN_MODE:
        for i, rollout in enumerate(rollouts):
            log_prob = dist_cat_t[i].log_prob(actions_t[i])
            rollout.add(batch_obs[i][0,:], actions_t[i][0], rewards_t[i], values_t[i][0][0], log_prob[0], 1 - dones_t[i])

    t += ACTORS
    
    for i, stack in enumerate(batch_obs):
        batch_obs[i] = obs2s_t[i]

    if LR_DECAY == 'linear':
        actorCritic.decay_clip_param(t)
        actorCritic.decay_learning_rate(t)
