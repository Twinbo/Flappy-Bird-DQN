# %% Load libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
# import mdock
from Flappy_Bird import CrappyBirdGame
import pygame

# %% Parameters
n_games = 2000
epsilon = 1
epsilon_min = 0.01
epsilon_reduction_factor = 0.01**(1/10000)
gamma = 0.99
batch_size = 512
buffer_size = 100_000
learning_rate = 0.0001
steps_per_gradient_update = 10
max_episode_step = 500
input_dimension = 3
hidden_dimension = 256
output_dimension = 2
save_interval = 100 #Gemmer vægtende for hvert 100 episode
max_score = -30

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
# Convert environment state to neural network input by one-hot encoding the state
def state_to_input(state):
    bird_y, bird_vel, dist_1, pipe_top_1, pipe_btm_1, dist_2, pipe_top_2, pipe_btm_2 = state

    max_bird_y = 768
    max_dist = 500
    max_gap_height = 768

    # Normalize each value to the range [-1, 1]
    bird_y_normalized = (2 * bird_y / max_bird_y) - 1
    bird_vel_normalized = bird_vel / 10
    dist_1_normalized = (2 * dist_1 / max_dist) - 1
    pipe_top_1_normalized = (2 * pipe_top_1 / max_gap_height) - 1
    pipe_btm_1_normalized = (2 * pipe_btm_1 / max_gap_height) - 1
    dist_2_normalized = (2 * dist_2 / max_dist) - 1
    pipe_top_2_normalized = (2 * pipe_top_2 / max_gap_height) - 1
    pipe_btm_2_normalized = (2 * pipe_btm_2 / max_gap_height) - 1

    # Create a tensor with normalized values
    return torch.hstack((
        torch.tensor([bird_y_normalized], dtype=torch.float32),
        torch.tensor([bird_vel_normalized], dtype=torch.float32),
        torch.tensor([dist_1_normalized], dtype=torch.float32),
        torch.tensor([pipe_top_1_normalized], dtype=torch.float32),
        torch.tensor([pipe_btm_1_normalized], dtype=torch.float32),
        torch.tensor([dist_2_normalized], dtype=torch.float32),
        torch.tensor([pipe_top_2_normalized], dtype=torch.float32),
        torch.tensor([pipe_btm_2_normalized], dtype=torch.float32)
    ))

# %% Environment
env = CrappyBirdGame()
action_names = env.actions        # Actions the environment expects
actions = np.arange(2)            # Action numbers


# %% Buffers
obs_buffer = torch.zeros((buffer_size, input_dimension))
obs_next_buffer = torch.zeros((buffer_size, input_dimension))
action_buffer = torch.zeros(buffer_size).long()
reward_buffer = torch.zeros(buffer_size)
done_buffer = torch.zeros(buffer_size)

# %% Training loop

# Logging
scores = []
losses = []
episode_steps = []
step_count = 0
print_interval = 100

# Training loop
for i in range(n_games):
    #print()
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
        #print(env_observation) 
        # Choose action and step environment
        if np.random.rand() < epsilon:
            # Random action
            action = np.random.choice(actions)
        else:
            # Action according to policy
            action = np.argmax(q_net(observation).detach().numpy())
        env_observation_next, reward, done = env.step(action)
        observation_next = state_to_input(env_observation_next)
        # if done:
        #     print('done')
        #print(reward)
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
            val = out[np.arange(batch_size), action_buffer[batch_idx]]   # Explain this indexing
            with torch.no_grad():
                out_next = q_net(obs_next_buffer[batch_idx])
                target = reward_buffer[batch_idx] + \
                    gamma*torch.max(out_next, dim=1).values * \
                    (1-done_buffer[batch_idx])
            loss = loss_function(val, target)


            # # Print statements for debugging
            # print(f"Batch indices: {batch_idx}")
            # print(f"Observations: {obs_buffer[batch_idx]}")
            # print(f"Next Observations: {obs_next_buffer[batch_idx]}")
            # print(f"Actions: {action_buffer[batch_idx]}")
            # print(f"Rewards: {reward_buffer[batch_idx]}")
            # print(f"Dones: {done_buffer[batch_idx]}")
            # print(f"Q-values: {out}")
            # print(f"Target values: {target}")
            # print(f"Loss: {loss.item()}")

            # Step the optimizer
            optimizer.zero_grad()
            loss.backward()
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
        # Print average score and number of steps in episode
        max_episode_score = np.max(scores[-print_interval:-1])
        average_score = np.mean(scores[-print_interval:-1])
        average_episode_steps = np.mean(episode_steps[-print_interval:-1])
        if max_episode_score > max_score:
            max_score = max_episode_score
        print(f'Episode={i+1}, Score={average_score:.1f}, Steps={average_episode_steps:.0f}, max episode sore={max_episode_score:.1f}, max socre {max_score}')
        # Plot scores        
        plt.figure('Score')
        plt.clf()
        plt.plot(scores, '.')
        plt.title(f'Step {step_count}: eps={epsilon:.3}')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.grid(True)
        
        # Plot number of steps
        plt.figure('Steps per episode')
        plt.clf()
        plt.plot(episode_steps, '.')
        plt.xlabel('Episode')
        plt.ylabel('Steps')
        plt.grid(True)

        # Plot last batch loss
        plt.figure('Loss')
        plt.clf()
        plt.plot(losses, '.')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.grid(True)

        # Estimate validation error

        #mdock.drawnow()

        # Save model
    if i % save_interval == 0:
        torch.save(q_net.state_dict(), 'q_net.pt')
        print(f"Model weights saved at episode {i}")

    #print(f'Episode={i+1}, Score={average_score:.1f}, Steps={average_episode_steps:.0f}')

    # Reduce epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_reduction_factor)

# Main function to choose training mode
# Main function to choose training mode
if __name__ == "__main__":
    mode = input("Enter 'new' to train from scratch or 'continue' to continue training: ").strip().lower()
    if mode == 'new':
        print("Training from scratch...")
        epsilon = 1  # Reset exploration rate for new training
        step_count = 0  # Reset step counter
        scores = []  # Clear scores
        losses = []  # Clear losses
        episode_steps = []  # Clear episode steps
    elif mode == 'continue':
        print("Loading trained model and continuing training...")
        try:
            q_net.load_state_dict(torch.load('q_net.pt'))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print("No saved model found. Starting from scratch...")
        # Optionally adjust epsilon if you want to balance exploration/exploitation
        epsilon = 0.1  # Lower exploration rate to leverage learned policy
    else:
        print("Invalid mode selected. Please enter 'new' or 'continue'.")
        exit()

    # Start training loop
    for i in range(n_games):
        score = 0
        episode_step = 0
        episode_loss = 0
        episode_gradient_step = 0
        done = False
        env_observation = env.reset_game()
        observation = state_to_input(env_observation)

        while not done and episode_step < max_episode_step:
            # Action selection
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)  # Random action
            else:
                action = np.argmax(q_net(observation).detach().numpy())  # Best action

            # Take action and observe outcome
            env_observation_next, reward, done = env.step(action)
            observation_next = state_to_input(env_observation_next)
            score += reward

            # Store experience
            buffer_index = step_count % buffer_size
            obs_buffer[buffer_index] = observation
            obs_next_buffer[buffer_index] = observation_next
            action_buffer[buffer_index] = action
            reward_buffer[buffer_index] = reward
            done_buffer[buffer_index] = done

            # Update observation
            observation = observation_next

            # Train on a batch from the replay buffer
            if step_count > batch_size and step_count % steps_per_gradient_update == 0:
                batch_idx = np.random.choice(
                    min(buffer_size, step_count), size=batch_size, replace=False
                )
                out = q_net(obs_buffer[batch_idx])
                val = out[np.arange(batch_size), action_buffer[batch_idx]]
                with torch.no_grad():
                    out_next = q_net(obs_next_buffer[batch_idx])
                    target = reward_buffer[batch_idx] + gamma * torch.max(out_next, dim=1).values * (1 - done_buffer[batch_idx])
                loss = loss_function(val, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_gradient_step += 1
                episode_loss += loss.item()

            episode_step += 1
            step_count += 1

        scores.append(score)
        losses.append(episode_loss / (episode_gradient_step + 1))
        episode_steps.append(episode_step)

        # Periodic logging and saving
        if (i + 1) % print_interval == 0:
            avg_score = np.mean(scores[-print_interval:])
            avg_steps = np.mean(episode_steps[-print_interval:])
            print(f"Episode={i+1}, Avg Score={avg_score:.2f}, Avg Steps={avg_steps:.0f}")
            plt.figure('Score')
            plt.clf()
            plt.plot(scores, '.')
            plt.title(f'Step {step_count}: eps={epsilon:.3}')
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.grid(True)
            
            # Plot number of steps
            plt.figure('Steps per episode')
            plt.clf()
            plt.plot(episode_steps, '.')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True)
 
            # Plot last batch loss
            plt.figure('Loss')
            plt.clf()
            plt.plot(losses, '.')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.show()

        if i % save_interval == 0:
            torch.save(q_net.state_dict(), 'q_net.pt')
            print(f"Model saved at episode {i + 1}")


# %%
