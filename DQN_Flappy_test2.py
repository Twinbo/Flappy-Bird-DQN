# %% Load libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from Flappy_Bird import CrappyBirdGame

# %% Parameters
n_games = 10000
epsilon = 1
epsilon_min = 0.01
epsilon_reduction_factor = 0.01**(1 / 10000)
gamma = 0.99
batch_size = 512
buffer_size = 100_000
learning_rate = 0.001
steps_per_gradient_update = 10
max_episode_step = 500
input_dimension = 3
hidden_dimension = 256
output_dimension = 2
save_interval = 100
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
def state_to_input(state):
    bird_y, bird_dist, gap_height = state

    max_bird_y = 551
    max_dist = 530
    max_gap_height = 551

    # Normalize each value between 0 and 1
    bird_y_normalized = bird_y / max_bird_y
    bird_dist_normalized = bird_dist / max_dist
    gap_height_normalized = gap_height / max_gap_height

    return torch.hstack((
        torch.tensor([bird_y_normalized], dtype=torch.float32),
        torch.tensor([bird_dist_normalized], dtype=torch.float32),
        torch.tensor([gap_height_normalized], dtype=torch.float32)
    ))

# %% Environment
env = CrappyBirdGame()
action_names = env.actions
actions = np.arange(2)

# %% Buffers
obs_buffer = torch.zeros((buffer_size, input_dimension))
obs_next_buffer = torch.zeros((buffer_size, input_dimension))
action_buffer = torch.zeros(buffer_size).long()
reward_buffer = torch.zeros(buffer_size)
done_buffer = torch.zeros(buffer_size)

# %% Main function
if __name__ == "__main__":
    mode = input("Enter 'new' to train from scratch or 'continue' to continue training: ").strip().lower()

    # Handle modes
    if mode == 'new':
        print("Training from scratch...")
        epsilon = 1  # Full exploration
        step_count = 0
        scores = []
        losses = []
        episode_steps = []
    elif mode == 'continue':
        print("Continuing from saved model...")
        try:
            q_net.load_state_dict(torch.load('q_net.pt'))
            print("Model weights loaded successfully.")
            print("Loaded weights:", list(q_net.parameters())[0])  # Debug weights
        except FileNotFoundError:
            print("No saved model found. Starting from scratch...")
        epsilon = 0.1  # Lower exploration to exploit trained policy
        optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate * 0.5)  # Lower learning rate
    else:
        print("Invalid mode selected. Please enter 'new' or 'continue'.")
        exit()

    # Logging
    scores = []
    losses = []
    episode_steps = []
    step_count = 0
    print_interval = 100

    # Training loop
    for i in range(n_games):
        # Reset episode
        score = 0
        episode_step = 0
        episode_loss = 0
        episode_gradient_step = 0
        done = False
        env_observation = env.reset_game()
        observation = state_to_input(env_observation)

        # Episode loop
        while not done and episode_step < max_episode_step:
            # Choose action
            if np.random.rand() < epsilon:
                action = np.random.choice(actions)  # Explore
            else:
                action = np.argmax(q_net(observation).detach().numpy())  # Exploit

            # Step environment
            env_observation_next, reward, done = env.step(action)
            observation_next = state_to_input(env_observation_next)
            score += reward

            # Store transition in buffer
            buffer_index = step_count % buffer_size
            obs_buffer[buffer_index] = observation
            obs_next_buffer[buffer_index] = observation_next
            action_buffer[buffer_index] = action
            reward_buffer[buffer_index] = reward
            done_buffer[buffer_index] = done

            # Update observation
            observation = observation_next

            # Learn every steps_per_gradient_update
            if step_count > batch_size and step_count % steps_per_gradient_update == 0:
                batch_idx = np.random.choice(
                    min(buffer_size, step_count), size=batch_size, replace=False)
                out = q_net(obs_buffer[batch_idx])
                val = out[np.arange(batch_size), action_buffer[batch_idx]]
                with torch.no_grad():
                    out_next = q_net(obs_next_buffer[batch_idx])
                    target = reward_buffer[batch_idx] + gamma * torch.max(out_next, dim=1).values * (1 - done_buffer[batch_idx])
                loss = loss_function(val, target)

                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                episode_gradient_step += 1
                episode_loss += loss.item()

            episode_step += 1
            step_count += 1

        # Logging metrics
        scores.append(score)
        losses.append(episode_loss / (episode_gradient_step + 1))
        episode_steps.append(episode_step)

        # Print and visualize
        if (i + 1) % print_interval == 0:
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

            # Plot episode steps
            plt.figure('Steps per episode')
            plt.clf()
            plt.plot(episode_steps, '.')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True)

            # Plot losses
            plt.figure('Loss')
            plt.clf()
            plt.plot(losses, '.')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.show()

        # Save model periodically
        if i % save_interval == 0:
            torch.save(q_net.state_dict(), 'q_net.pt')
            print(f"Model saved at episode {i + 1}")

        # Reduce epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_reduction_factor)

# %%
