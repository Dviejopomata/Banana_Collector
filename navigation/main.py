from collections import deque
import numpy as np
import torch
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment

from navigation.agent import Agent
from navigation.constants import WEIGHTS_FILE


def train_dqn(env, agent,
              solved_cond=14.0, n_episodes=2000, max_t=1000,
              eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Train a Deep Q-Learning agent in the given environment.
    Args
    ====
        env (object): Unity environment; the function uses the default brain
        agent (object): RL agent
    Params
    ======
        solved_cond (float): averaged score condition to consider the environment solved; average is over 100 episodes
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    brain_name = env.brain_names[0]  # get the default brain
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        state = env_info.vector_observations[0]
        score = 0
        for t in range(max_t):
            action = int(agent.act(state, eps))
            env_info = env.step(action)[brain_name]  # send the action to the environment
            next_state = env_info.vector_observations[0]  # get the next state
            reward = env_info.rewards[0]  # get the reward
            done = env_info.local_done[0]  # check if the episode is done
            agent.update(state, action, reward, next_state, done)  # update agent Q-values and replay buffer
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)  # save most recent score
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        # Check if the averaged score is larger/equal than the condition for solving the environment
        if np.mean(scores_window) >= solved_cond:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), WEIGHTS_FILE)
            break
    return scores


env_name = "./Banana_Linux/Banana.x86_64"

env = UnityEnvironment(file_name=env_name)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)
state_size = len(env_info.vector_observations[0])
print('Number of states:', state_size)

LOAD_MODEL = False
# Define agent
agent = Agent(state_size=state_size, action_size=action_size, neurons=50, seed=0)

if LOAD_MODEL:
    agent.qnetwork_local.load_state_dict(torch.load(WEIGHTS_FILE))

# reset the environment

# Train agent
scores = train_dqn(env, agent)

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

# Show agent sampling correct bananas
print("\n Testing agent's performance...")
env_info = env.reset(train_mode=False)[brain_name]  # reset the environment
state = env_info.vector_observations[0]  # get the current state

score = 0  # initialize the score
eps = 0.01

while True:
    action = int(agent.act(state, eps))  # select an action
    env_info = env.step(action)[brain_name]  # send the action to the environment
    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0]  # get the reward
    done = env_info.local_done[0]  # see if episode has finished
    score += reward  # update the score
    state = next_state  # roll over the state to next time step
    if done:  # exit loop if episode finished
        print('Done!')
        print("Score: {}".format(score))
        break

env.close()
