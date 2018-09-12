import gym
import os
import sys
from q_learner import QLearner
import matplotlib.pyplot as plt
import shutil
import numpy as np
import pdb

"""
Usage:
python run_game.py [name=<env name>,param1_name=<param1>,param2_name=<param2>...] <file directory to save results> <optional: learned Q-value file>
This will train a tabular Q-learning agent on the provided environment (with given parameters) and save the results into the provided directory.
If a learned Q-value file is provided, you can watch the agent play the game with the learned Q-value function.
e.g. python run_game.py [name="TargetCatcher-v0",blindspot_region="250;300;350;400;450"] save_dir target_task_learnedQ/Q.csv
"""

def run(env, agent, render, save_dir, num_episodes):
    max_timesteps = 100
    save_freq = 1000 # Q-values and plot will be updated at this frequency
    interval = 10000 # Reward plot will only plot rewards at this frequency
    all_rewards = []
    mean_rewards = []
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    for i_episode in range(num_episodes):
        total_reward = 0
        state = env.reset()
        done = False
        for t in range(max_timesteps):
            if render:
                env.render()
            action = agent.select_action(state)
            next_state, reward, done, _info = env.step(action)
            # print(state, action, next_state, reward)
            # input()
            agent.updateQ(state, action, reward, next_state)
            total_reward += reward
            state = next_state
            if done:
                break
        # print(total_reward)
        all_rewards.append(total_reward)
        # Keep track of mean reward over last 100 episodes (smoother learning curve)
        mean_rewards.append(float(np.mean(all_rewards[-100:])))
        if i_episode % save_freq == 0:
            x = range(i_episode+1)[::interval]
            mean_rewards_y = mean_rewards[::interval]

            agent.saveQ(save_dir)
            agent.save_debug_info(save_dir)

            fig2 = plt.figure(1)
            ax2 = fig2.add_subplot(1,1,1)
            ax2.clear()
            ax2.plot(x, mean_rewards_y)
            plt.savefig(os.path.join(save_dir, "mean_rewards.png"))

if __name__ == "__main__":
    # Env name to train on (e.g., Catcher-v0) and additional parameters
    env_info = sys.argv[1][1:-1].split(',')
    env_params = {}
    for s in env_info:
        pair = s.split('=')
        if pair[0] == "name":
            env_name = pair[1]
        else:
            env_params[pair[0]] = pair[1]
    env = gym.make(env_name)
    if hasattr(env.env, "game"):
        env.env.game.params = env_params

    # If a learned Q-value file is given, the game will be rendered, and the agent will play according to the learned Q-value function.
    if len(sys.argv) >= 4:
        agent = QLearner(env, sourceQ_file=sys.argv[3])
        render = True
    else:
        agent = QLearner(env)
        render = False

    num_episodes = 10000000
    save_dir = sys.argv[2]

    run(env, agent, render, save_dir, num_episodes)