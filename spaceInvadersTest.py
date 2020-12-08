import tensorflow as tf
import numpy as np
import gym

from skimage import transform
from skimage.color import rgb2gray

import matplotlib.pyplot as plt
from collections import deque

import random
import warnings

warnings.filterwarnings('ignore')



# Create environment and analyze: ---------------------------------------------------------------------------------------------------------------------------------------------------

env = gym.make('SpaceInvaders-v0')

observationSpace = env.observation_space
action_size = env.action_space.n


# print("Size of frame: ", env.observation_space)
# print("Size of actions: ", env.action_space.n)

print("Size of frame: ", observationSpace)
print("Size of actions: ", action_size)


# Hot encoded version of actions:
possible_actions = np.array(np.identity(action_size,dtype=int).tolist())


# PreProcess frame: ---------------------------------------------------------------------------------------------------------------------------------------------------------------

def preProcesseFrame(frame):
    # Turn frame to grayscale
    grayFrame = rgb2gray(frame)

    # Crop the screen (remove the part below the player)
    # [Up, Down, Left, Right]
    cropped_frame = grayFrame[8:-12,4:-12]
    normalized_frame = cropped_frame/255.0

    # Resize the frame
    preProcessedFrame = transform.resize(cropped_frame, [110,84])

    return preProcessedFrame

# Create Stack of frames: ----------------------------------------------------------------------------------------------------------------------------------------------------------
stack_size = 4                                                                                                      # 4 Frames stack.
stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)],maxlen=4)                      # Initialize deque with zero-images one array for each image.

def stackFrames(stacked_frames, state, isNewEpisode):
    # Run preprocess for frame
    frame = preProcesseFrame(state)

    if isNewEpisode:
        # Clear stacked frames
        stacked_frames = deque([np.zeros((110,84), dtype=np.int) for i in range(stack_size)],maxlen=4)

        # Copy the same frame 4 times in new episode to initialize
        for i in range(4):
            stacked_frames.append(frame)
        
        # Stack the frames
        stacked_state = np.stack(stacked_frames, axis=2)

    else:
        # Append frame to deque [automatically removes oldest frame]
        stacked_frames.append(frame)

        # Build the stacked state [first dimension specifies different frames]
        stacked_state = np.stack(stacked_frames, axis=2)

    return stacked_state, stacked_frames


# Model HYPERPARAMETERS:
state_size = [110, 84, 4]               # 4 frames stack [110x84x4 width,height, channels]
learning_Rate = 0.00025                 # Learning rate (ALPHA)

# Training HYPERAPARAMETERS:
total_episodes = 50                     # Total episodes for training
max_steps = 50000                       # Max posible steps in an episode
batch_size = 64                         # Batch size

# Exploration parameters for epsilon greedy stratey
explore_start = 1.0                     # Exploration probability at start
explore_stop = 0.01                     # Minimum exploration probability
decay_rate = 0.00001                    # Exponential decay rate for exploration probability

# Q learning HYPERPARAMETERS
gamma = 0.9                             # Discounting rate

# Memory HYPERPARAMETERS
pretrain_length = batch_size            # Number of experiences stored in the memorwy when intialized for the first time
memory_size = 1000000                   # Number of experiences the memory can keep

# Preprocessing HYPERAPARAMETERS
stack_size = 4                          # Number of frames stacked


# OPTIONS FOR VIEWING AGENTS!
training = False
renderEpisode = False



# AGENT DQNetwork: ----------------------------------------------------------------------------------------------------------------------------------------------------------
class DQNetwork:
    def __init__(self, state_size, action_size, learning_Rate, name='DQNetwork'):
        self.state_size = state_size
        self.actions_size = action_size
        self.learning_Rate = learning_Rate

        with tf.compat.v1.variable_scope(name):

            # Create the placeholders
            # Take each element of state_size in tuple like [None, 110, 84, 4]

            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *state_size], name="inputs")
            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, *action_size], name="actions")


            # WEIRD FORMULA I DONT UNDERSTAND:
            self.target_Q = tf.compat.v1.placeholder(tf.float32, [None], name="target")


            # Input is 110 x 84 x 4
            # FIRST CONVNET
            self.conv1 = tf.compat.v1.layers.conv2d(inputs = self.inputs_, filters =32, kernel_size = [8,8], strides = [4,4], padding = "VALID", kernel_initializer = tf.conrib.layers.xavier_initializer_conv2d(), name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            # SECOND CONVNET
            self.conv2 = tf.compat.v1.layers.conv2d(inputs = self.conv1_out, filters =64, kernel_size = [4,4], strides = [2,2], padding = "VALID", kernel_initializer = tf.conrib.layers.xavier_initializer_conv2d(), name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            # THIRDS CONVNET
            self.conv3 = tf.compat.v1.layers.conv2d(inputs = self.conv2_out, filters =64, kernel_size = [3,3], strides = [2,2], padding = "VALID", kernel_initializer = tf.conrib.layers.xavier_initializer_conv2d(), name="conv3")
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.contrib.layers.flatten(self.conv3_out)

            self.fc = tf.compat.v1.layers.dense(inputs = self.flatten, units=512, activation = tf.nn.elu, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), name="fcl")
            self.output =tf.compat.v1.layers.dense(inputs = self.fc, kernel_initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform", units = self.actionSize, activation = None)) 

            # Q is the predicted value
            self.Q = tf.reduce_sum(input_tensor=tf.multiply(self_output, actions_))

            # The loss is the difference between our predicted Q_values and the Q_target
            self.loss = tf.reduce_mean(input_tensor=tf.square(self.target_Q - self.Q))

            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.training_rate).minimize(self.loss)




# Reset the graph
tf.compat.v1.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = DQNetwork(state_size, action_size, learning_Rate)


# Replay method to create experience: ----------------------------------------------------------------------------------------------------------------------------------------------


# CREATE MEMORY:
class Memory():
    def __init__(self, max_size):
        self.buffer = deque( maxlen= max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        buffer_size = len(self.buffer)
        index = np.random.choice(np.arrange(buffer_size), size=batch_size, replace=False)

        return [self.buffer[i] for i in index]
            
# INSTANTIATE MEMORY:

memory = Memory(max_size = memory_size)
for i in range(pretrain_length):

    # IF its the first step
    if i == 0:
        state = env.reset()

        state, stacked_frames = stackFrames(stacked_frames, state, True)

    # Get the next state, rewards donde by a random action
    choice = random.randint(1,len(possible_actions)) -1
    action = possible_actions[choice]
    next_state, reward, done, _ = env.step(action)

    
    # Stack the frames
    next_state, stacked_frames = stackFrames(stacked_frames, next_state, False)

    # If episode is finished (Ai DIED)

    if done:
        
        # Episode is finished
        next_state = np.zeros(state.shape)

        # Add experience to memory
        memory.add((state, action, reward, next_state, done))

        # Start a new episode
        state = env.reset()

        # Stack the frames
        state, stacked_frames = stack_frames(stacked_frames, state, True)

    else:

        # Add experience to memory
        memory.add((state, action, reward, next_state, donde))

        # New state becomes next state
        state = next_state


# TENSORBOARD WRITER: -----------------------------------------------------------------------------------------------------------------------------------------------------
writer = tf.compat.v1.summary.FileWriter("/tensorboard/dqn/1")

# Losses
tf.compat.v1.summary.scalar("Loss", DQNetwork.loss)

write_op = tf.compat.v1.summary.merge_all()


# TRAIN THE AI: --------------------------------------------------------------------------------------------------------------------------------------------------------------

def predictionAction(explore_start, explore_stop, decay_rate, decay_step, state, actions):
    # EPSILON GREEDY STRRATEGY

    # Choose state using epsilon greedy strategy
    # Randomize number
    exp_exp_tradeoff = np.random.rand()

    explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

    if (explore_probability > exp_exp_tradeoff):
        # Make random action
        choice = random.randint(1,len(possible_actions)) -1
        action = possible_actions[choice]

    else:
        # Get action from Q-Network
        # Estimate the Qs values state
        Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state.reshape((1, *state.shape))})

        # Take the biggest Q value (= the best_action)
        choice = np.argmax(Qs)
        action = possible_actions[choice]

    return action, explore_probability

# Saver saves the model:

saver = tf.compat.v1.train.Saver()

if training == True:
    with tf.compat.v1.Session() as sess:

        sess.run(tf.compat.v1.global_variables_initializer())

        # Initialize decay rate to reduce epsilon
        decay_step = 0

        for episodes in range (total_episodes):
            # Step to 0
            step = 0

            # Rewards of episode
            episode_rewards = []

            # Make new episode and observe the first state
            state = env.reset()

            # Remember that stack frame function also call our preprocess function.
            state, stacked_frames = stackFrames(stacked_frames, state, True)

            while step < max_steps:
                step += 1

                # Increase decay step
                decay_step += 1

                # Predict the action to take and take it
                action, explore_probability = predictionAction(explore_start, explore_stop, decay_rate, decay_step, state, possible_actions)

                # Perform the action and get the next_state, reward, and done information
                next_state, reward, done, _ =env.step(action)

                if episode_render:
                    env_render()

                # Add the reward to total rewards:
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # The episode ends so no next state:
                    next_state = np.zeros((110,84), dtype=np.int)

                    next_state, stacked_frames = stackFrames(stacked_frames, next_state, False)

                    # Set step = max_steps to end the episode
                    step = max_steps

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode), 'Total reward: {}'.format(total_reward),'Explore P: {:.4f}'.format(explore_probability), 'Training loss {:.4f}'.format(loss))

                    rewards_list.append((episode,total_reward))

                    # Store transition <st,at,rt+1,st+1> in memory D
                    memory.add((state, action, reward, next_state, done))

                else:
                    # Stack the frame of the next step:
                    next_state, stacked_frames = stackFrames(stacked_frames, next_state, False)

                    # Add experience to memory:
                    memory.add((state, action, reward, next_state, done))

                    #  st+1 is now our current state
                    state = next_state


                    # Learning part: ---------------------------------------------------------------------------------------------------------------------------------------------------
                    # Random mini batch for memory
                    batch = memory.sample(batch_size)
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch])
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    DONES_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []

                    # Get Q values for next_state
                    target_Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: next_states_mb})

                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])
                        else:
                            target = rewards_mb[i] + gamma * np.max(target_Qs[i])
                            target_Qs_batch.append(target)

                        targets_mb = np.array([each for each in target_Qs_batch])

                        loss, _ = sess.run ([DQNetwork.loss, DQNetwork.optimizer], feed_dict={DQNetwork.inputs_: states_mb, DQNetwork.target_Q: targets_mb, DQNetwork.actions_: actions_mb})

                        # Write TF summaries

                        summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb, DQNetwork.target_Q: targets_mb, DQNetwork.actions_: actions_mb})

                        wirter.add_summary(summary, episode)
                        writer.flush()

                        # Save model every 5 episodes
                        if episode % 5 == 0:
                            save_path = saver.save(sess, "./models/model.ckpt")
                            print("model saved")
                    

with tf.compat.v1.Session() as sess:
    total_test_rewards = []

    # Load the model

    saver.restore(sess, "./models/model.ckpt")

    for episode in range(1):
        total_rewards = 0
        state = env.reset()
        
        state, stacked_frames = stackFrames(stacked_frames, state, True)

        print("--------------------------------------------------------------")
        print("Episode", episode)

        while True:
            # Reshape the state
            state = state.reshape((1, *state_size))

            # Get action from Q-Network
            # Estimate the Qs value state

            Qs = sess.run(DQNetwork.output, feed_dict = {DQNetwork.inputs_: state})

            # Take the biggest Q value
            choice = np.argmax(Qs)
            actino = possible_actions[choice]

            # Perform the action and get the next state, reward, and done information
            next_state, reward, done, _ = env.step(action)
            env.render()

            total_rewards += reward

            if done:
                print("Score", total_rewards)
                total_test_rewards.append(total_rewards)
                break

            next_state, stacked_frames = stackFrames(stacked_frames, next_state, False)
            state = next_state

    env.close()