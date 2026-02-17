#Denne kode er blevet inspireret af vores underviseres eksempel, som vi har videre bygget på og tilpasset til vores eget projekt.


# %% Load libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from Flappybird import FlappyBirdGame
import pygame
import os

# %% Parameter
n_games = 1000
epsilon = 1
epsilon_min = 0.001
#epsilon_reduction_factor = 0.001**(1/15000)
# epsilon_reduction_factor = 0.99995
epsilon_reduction_factor = 0.9995
gamma = 0.99
batch_size = 512
buffer_size = 750000
learning_rate = 0.0001
steps_per_gradient_update = 10
max_episode_step = 8000
input_dimension = 5
hidden_dimension = 256
output_dimension = 2
max_score = -40


# %% Neural network, optimizer and loss
q_net = torch.nn.Sequential(
    torch.nn.Linear(input_dimension, hidden_dimension),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dimension, hidden_dimension),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden_dimension, output_dimension)
)
optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)
loss_function = torch.nn.MSELoss()

# %% State to input transformation
# Convert environment state to neural network input 
def state_to_input(state):
    bird_y, bird_vel, bird_dist, pipe_top, pipe_btm = state

    max_bird_y = 768
    max_dist = 400
    max_gap_height = 768


    # Normalize each value to the range [-1, 1]
    bird_y_normalized = (2 * bird_y / max_bird_y) - 1  
    bird_vel_normalized = bird_vel / 10  
    bird_dist_normalized = (2 * bird_dist / max_dist) - 1
    pipe_top_normal = (2 * pipe_top / max_gap_height) - 1
    pipe_btm_normal = (2 * pipe_btm / max_gap_height) - 1

     # Create a tensor with normalized values
    return torch.hstack((
        torch.tensor([bird_y_normalized], dtype=torch.float32),
        torch.tensor([bird_vel_normalized], dtype=torch.float32),
        torch.tensor([bird_dist_normalized], dtype=torch.float32),
        torch.tensor([pipe_top_normal], dtype=torch.float32),
        torch.tensor([pipe_btm_normal], dtype=torch.float32)
    ))

# %% Load the trained model from the checkpoint
checkpoint_path = 'q_net_checkpoint.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    q_net.load_state_dict(checkpoint['model_state_dict'])
    print("Trained model loaded for testing.")
else:
    raise FileNotFoundError("Checkpoint not found. Train the model first.")


# %% Environment
env = FlappyBirdGame()
action_names = env.actions        # Actions the environment expects
actions = np.arange(2)            # Action numbers


# %% Buffers
obs_buffer = torch.zeros((buffer_size, input_dimension))
obs_next_buffer = torch.zeros((buffer_size, input_dimension))
action_buffer = torch.zeros(buffer_size).long()
reward_buffer = torch.zeros(buffer_size)
done_buffer = torch.zeros(buffer_size)


# %% check if data is available
# Load checkpoint if it exists
checkpoint_path = 'q_net_checkpoint_0_0001 training 20000.pt'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    q_net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    step_count = checkpoint['step_count']
    epsilon = checkpoint['epsilon']
    print(f"Checkpoint loaded. Resuming training from Step {step_count}, Epsilon={epsilon:.4f}.")
else:
    step_count = 0
    epsilon = 1
    print("No checkpoint found. Starting training from scratch.")

# %% Training loop
# Logging
scores = []
losses = []
episode_steps = []
step_count = 0
print_interval = 100

# Training loop
for i in range(n_games):

    # Reset game
    score = 0
    episode_step = 0
    episode_loss = 0
    episode_gradient_step = 0
    done = False
    env_observation = env.reset_game()
    observation = state_to_input(env_observation)

    # Reduce exploration rate
    epsilon = (epsilon-epsilon_min)*epsilon_reduction_factor + epsilon_min
    
    # Episode loop
    while (not done) and (episode_step < max_episode_step): 
        # Choose action and step environment
        if np.random.rand() < epsilon:
            # Random action
            action = np.random.choice(actions)
        else:
            # Action according to policy
            action = np.argmax(q_net(observation).detach().numpy())
        env_observation_next, reward, done = env.step(action)
        observation_next = state_to_input(env_observation_next)

        score += reward

        # Store to buffers
        buffer_index = step_count % buffer_size
        obs_buffer[buffer_index] = observation
        obs_next_buffer[buffer_index] = observation_next
        action_buffer[buffer_index] = action
        reward_buffer[buffer_index] = reward
        done_buffer[buffer_index] = done

        # Update to next observation
        observation = observation_next

        # Learn using minibatch from buffer (every steps_per_gradient_update)
        if step_count > batch_size and step_count%steps_per_gradient_update==0:
            # Choose a minibatch            
            batch_idx = np.random.choice(np.minimum(
                buffer_size, step_count), size=batch_size, replace=False)

            # Compute loss function
            out = q_net(obs_buffer[batch_idx])
            val = out[np.arange(batch_size), action_buffer[batch_idx]]  
            with torch.no_grad():
                out_next = q_net(obs_next_buffer[batch_idx])
                target = reward_buffer[batch_idx] + \
                    gamma*torch.max(out_next, dim=1).values * \
                    (1-done_buffer[batch_idx])
            loss = loss_function(val, target)

            # Step the optimizer
            optimizer.zero_grad()
            loss.backward()

            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=20)

            optimizer.step()

            episode_gradient_step += 1
            episode_loss += loss.item()

        # Update step counteres
        episode_step += 1
        step_count += 1

    scores.append(score)
    losses.append(episode_loss / (episode_gradient_step+1))
    episode_steps.append(episode_step)
    if (i+1) % print_interval == 0:
        std_score = np.std(scores)
        mean_score = np.mean(scores)
        average_score = np.mean(scores[-print_interval:-1])
        average_episode_steps = np.mean(episode_steps[-print_interval:-1])
        if np.max(scores[-print_interval:-1]) > max_score:
            max_score = np.max(scores[-print_interval:-1])
        
        # Plot number of steps
        plt.figure('Steps per episode')
        plt.clf()
        plt.plot(episode_steps, '.')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)

        # Plot average scores
        average_score_window = 100 
        if len(scores) >= average_score_window:
            avg_scores = np.convolve(scores, np.ones(average_score_window) / average_score_window, mode='valid')
            plt.figure('Average Score')
            plt.clf()
            plt.plot(range(len(avg_scores)), avg_scores, label=f'Average Score)', color='orange')
            plt.title(f'Step {step_count}: eps={epsilon:.3}')
            plt.xlabel('Episode')
            plt.ylabel('Average Score')
            plt.grid(True)
            plt.legend()

        # Plot scores        
        plt.figure('Score')
        plt.clf()
        plt.plot(scores, '.')
        plt.title(f'Step {step_count}: eps={epsilon:.3}')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)

        # Plot last batch loss
        plt.figure('Loss')
        plt.clf()
        plt.plot(losses, '.')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)

        plt.pause(0.01)
        print(f'Episode={i+1}, Score={average_score:.1f}, mean={mean_score}, Steps={average_episode_steps:.0f}, max_score={max_score:.1f}, std_score={std_score:.1f}')

        # Save model and optimizer state as checkpoint
        torch.save({
            'model_state_dict': q_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'step_count': step_count,
            'epsilon': epsilon
        }, 'q_net_checkpoint.pt')

env.close()


# %%
