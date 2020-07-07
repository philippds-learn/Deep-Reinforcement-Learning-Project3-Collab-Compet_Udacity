[//]: # (Image References)

[image1]: images/ddpg-algorithm.JPG "A2C DDPG"
[image2]: images/ddpg-algorithm-intuition.JPG "A2C DDPG INTUITIVE"
[image3]: images/score.JPG "plot of rewards per episode"
[image4]: images/plot.JPG "plot"


## Deep Reinforcement Learning Continuous Control Project Submission Report

### Overview

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Learning Algorithm

#### Advantage Actor-Critic (A2C): Multi Agent Deep Deterministic Policy Gradient, Continuous Action-space (MA-DDPG)

![A2C DDPG][image1]

Two Deep Neural Networks, on the left hand side, the "actor" DNN on the right hand side the "critic" DNN.
The "actor" DNN's output is a approximate maximiser which is passed to the "critic" DNN's output to calculate a new target value for training the action value function ("much in the way DQN does").

![A2C DDPG INTUITIVE][image2]

DDPG Paper:
[click here](https://arxiv.org/abs/1509.02971)

More about A3C Q-Prop:
[click here](https://arxiv.org/abs/1611.02247)

#### Hyperparameters used:

##### For ddpg function:
n_episodes (int): maximum number of training episodes = 1000<br />

##### For ddpg Agent:
BUFFER_SIZE (int): replay buffer size = int(1e7)<br />
BATCH_SIZE (int): minibatch size = 128<br />
GAMMA (float): discount factor = 0.999<br />
TAU (int): for soft update of target parameters = 1e-2<br />
LR_ACTOR (int): learning rate of the actor = 1e-4<br />
LR_CRITIC (int): learning rate of the critic = 1e-3<br />
WEIGHT_DECAY (float): L2 weight decay = 0<br />

#### Actor (Policy) Model.
fcs1_units (int): Number of nodes in the first hidden layer = 256<br />
fc2_units (int): Number of nodes in the second hidden layer = 256<br />

#### Critic (Value) Model.
fcs1_units (int): Number of nodes in the first hidden layer = 256<br />
fc2_units (int): Number of nodes in the second hidden layer = 256<br />

### Plot of Rewards

![plot of rewards per episode][image3]
![plot][image4]

### Ideas for Future Work

#### 1. Amend the various hyperparameters
Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster.

#### 1.1 Automate hyperparameter variations
You could use for example a genetic algorithm to find out what hyperparameters result in desired improvements.

#### 2. Implementing different learning algorithm
You may like to implement [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf) (implemented), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb).
