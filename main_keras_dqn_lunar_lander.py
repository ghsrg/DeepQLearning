from simple_dqn_keras import Agent
import numpy as np
import gym
from utils import plotLearning
from termcolor import colored
from gym import wrappers

if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    n_games = 300
    agent = Agent(gamma=0.99, epsilon=0.19, alpha=0.0001, input_dims=8, epsilon_dec=0.997,
                  n_actions=4, mem_size=1000000, batch_size=64, epsilon_end=0.03)

    agent.load_model()
    scores = []
    eps_history = []

    #env = wrappers.Monitor(env, "tmp/lunar-lander-6",
    #                         video_callable=lambda episode_id: True, force=True)

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        i_try = 0
        while not done:
            i_try+=1
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            #env.render()
            if i_try > 1000:
                reward = -40
                done=1
            score += reward
            print(colored('Game # ' + str(i) + '/' + str(i_try), 'green'), colored(' Revard=' + str(reward), 'red'))
            agent.remember(observation, action, reward, observation_, int(done))
            observation = observation_
            agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[max(0, i-100):(i+1)])
        print('episode: ', i,'score: %.2f' % score,
              ' average score %.2f' % avg_score)

        if (i % 10 == 0 and i > 0) or i < 10:
            print('Svave model')
            agent.save_model()

        filename = 'lunarlander.png'

        x = [y+1 for y in range(i+1)]
        plotLearning(x, scores, eps_history, filename)

