from ple.games.flappybird import FlappyBird
from ple import PLE
import matplotlib.pyplot as plt
import pickle
import datetime

from MCAgent import MCAgent
from QLAgent import QLAgent
from LinearApproxAttempt import LinearApproxAttempt
from QLBestAgent import QLBestAgent

def train_agent(nb_frames, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = agent.reward_values())
    env.init()

    frame_counter = 0
    eval_scores = []
    eval_episodes = []

    while frame_counter <= nb_frames:
        # pick an action
        action = agent.training_policy(env.game.getGameState())
        if frame_counter % 10000 == 0:
            print("Agent has been trained on {} frames. Now evaluating".format(frame_counter))
            eval_scores.append(evaluate_agent(10, agent, frame_counter))
            eval_episodes.append(frame_counter)
            print("Average score was {}".format(eval_scores[-1]))
        # limit max frames at 1 million
        if frame_counter == 1000000:
            print("At a million frames breaking.")
            break
        # step the environment
        prevState = env.game.getGameState()
        reward = env.act(env.getActionSet()[action])
        agent.observe(prevState, action, reward, env.game.getGameState(), env.game_over())
        frame_counter += 1
        # reset the environment if the game is over
        if env.game_over():
            env.reset_game()

    plt.plot(eval_episodes, eval_scores)
    plt.draw()

def evaluate_agent(nb_runs, agent, frame_counter=0):
    reward_values = {"positive": 1.0, "loss":0.0}
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()
    scores = []
    score  = 0
    highest_score = -1
    while nb_runs:
        # pick an action
        action = agent.policy(env.game.getGameState())
        # step the environment
        reward = env.act(env.getActionSet()[action])
        score += reward
        # reset the environment if the game is over
        if env.game_over() or score >= 1000:
            scores.append(score)
            env.reset_game()
            nb_runs -= 1
            highest_score = max(highest_score, score)
            print("Score for evaluation: {}".format(score))
            score = 0
    print("Highest score for evaluation: {}".format(highest_score))
    print("States visited: {}".format(len(agent.q.keys())))
    return sum(scores)/len(scores)


def test_trained_agent(nb_runs, agent):
    reward_values = {"positive": 1.0, "loss": 0.0}
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()
    scores = []
    score  = 0
    highest_score = -1
    while nb_runs:
        # pick an action
        action = agent.policy(env.game.getGameState())
        # step the environment
        reward = env.act(env.getActionSet()[action])
        score += reward
        # reset the environment if the game is over
        if env.game_over():
            scores.append(score)
            highest_score = max(highest_score, score)
            print("Got score {} for this test".format(score))
            env.reset_game()
            nb_runs -= 1
            score = 0
    return

qlbest = QLBestAgent(0.1,1.0,0.1)
train_agent(1000000, qlbest)
test_trained_agent(5,qlbest)