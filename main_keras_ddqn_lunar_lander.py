#import os
# for keras the CUDA commands must come before importing the keras libraries
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import gym
from gym import wrappers
import pickle
import numpy as np
from ddqn_keras import DDQNAgent
from utils import plotLearning
from termcolor import colored


def save_game(f_game_name,i, ddqn_scores, eps_history,ddqn_avg_score):
    with open(f_game_name, "wb") as f:
        pickle.dump((i, ddqn_scores,eps_history,ddqn_avg_score), f)


def load_game(f_game_name):
    with open(f_game_name, "rb") as f:
        i, ddqn_scores, eps_history,ddqn_avg_score = pickle.load(f)
    return i, ddqn_scores, eps_history,ddqn_avg_score

if __name__ == '__main__':
    env = gym.make('LunarLander-v2',continuous=False)
    #env = gym.make('CarRacing-v2', continuous=False)
    n_games = 5000
    action_size = 4
    state_size = 8
    ddqn_agent = DDQNAgent(alpha=0.001, gamma=0.99, n_actions=action_size, input_dims=state_size, epsilon=1.0,
                  batch_size=64,  epsilon_dec=0.9997,  epsilon_end=0.01,
                  mem_size=100000, fname='dqn_model2', replace_target=100)

    ddqn_scores = []
    eps_history = []
    ddqn_avg_score = []
    #env = wrappers.Monitor(env, "tmp/lunar-lander-ddqn-2",
    #                         video_callable=lambda episode_id: True, force=True)
    i_l = 0

    i_l, ddqn_scores, eps_history, ddqn_avg_score = load_game('ddqn_game.txt')
    ddqn_agent.load_model()

    for i in range(i_l+1, n_games):
        done = False
        score = 0
        observation = env.reset()
        i_try = 0
        while not done:
            i_try += 1
            action = ddqn_agent.choose_action(observation)
            if i_try > 300:
                action = 0
            observation_, reward, done, info = env.step(action)

            #env.render()
            score += reward
            #print(colored('Game # ' + str(i) + '/' + str(i_try), 'green'), colored(' Revard=' + str(reward), 'red'))
            ddqn_agent.remember(observation, action, reward, observation_, int(done))
            ddqn_agent.learn(i_state=observation, new_state=observation_, i_action=action, reward = reward, done =done )
            observation = observation_
        eps_history.append(ddqn_agent.epsilon)
        #print('score: %.2f' % score)
        ddqn_agent.update_network_parameters()
        ddqn_scores.append(score)

        avg_score = np.mean(ddqn_scores[max(0, i-100):(i)])
        ddqn_avg_score.append(avg_score)
        print('episode:', i,', score: %.2f' % score,
              ', average score %.2f' % avg_score)

        if (i % 10 == 0 and i > 0) or i < 10:
            ddqn_agent.save_model()
        filename = 'lunarlander-ddqn.png'

        x = [y + 1 for y in range(i)]
        plotLearning(x, ddqn_avg_score, ddqn_scores, filename)
        save_game('ddqn_game.txt', i, ddqn_scores, eps_history, ddqn_avg_score)

