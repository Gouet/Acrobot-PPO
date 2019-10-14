import gym

class EnvWrapper:
    def __init__(self, gym_env, actors, update_obs=None, update_reward=None, end_episode=None):
        self.envs = []
        self.variables = []
        self.update_obs = update_obs
        self.episode = 0
        self.end_episode = end_episode
        self.update_reward = update_reward
        self.global_step = 0
        self.episode_step = []
        for _ in range(actors):
            self.envs.append(gym.make(gym_env))
        for _ in range(actors):
            self.variables.append([])
            self.episode_step.append(0)

    def add_variables_at_index(self, id, data):
        self.variables[id] = data

    def get_variables_at_index(self, id):
        return self.variables[id]

    def step(self, actions):
        batch_states = []
        batch_rewards = []
        batch_dones = []

        for i, action in enumerate(actions):
            self.episode_step[i] += 1
            states, rewards, done_, _ = self.envs[i].step(action)
            if done_ == True:
                states = self.envs[i].reset()
                self.episode += 1
                if self.end_episode is not None:
                    self.end_episode(self.episode, self.variables[i], self.global_step, self.episode_step[i])
                self.episode_step[i] = 0
                self.variables[i] = []
            if self.update_reward is not None:
                rewards = self.update_reward(rewards)
            if self.update_obs is not None:
                states = self.update_obs(states)
            batch_states.append(states)
            batch_rewards.append(rewards)
            batch_dones.append(done_)
        self.dones = batch_dones
        self.global_step += 1
        return batch_states, batch_rewards, batch_dones

    def render(self, id):
        self.envs[id].render()

    def done(self):
        return all(self.dones)

    def reset(self):
        batch_states = []
        self.dones = []
        print('RESET')
        for env in self.envs:
            obs = env.reset()
            self.dones.append(False)
            if self.update_obs is not None:
                obs = self.update_obs(obs)
            batch_states.append(obs)
        return batch_states