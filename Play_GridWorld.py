# Grid World: Manual play

# Instructions:
#   Move up, down, left, or right to move the character. The 
#   objective is to find the key and get to the door
#
# Control:
#    arrows  : Merge up, down, left, or right
#    r       : Restart game
#    q / ESC : Quit

from GridWorld import GridWorld
import pygame
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import torch
import copy

# %% Helper functions
def drawnow():
    plt.gcf().canvas.draw()
    plt.gcf().canvas.flush_events()

# Enable interactive plotting
plt.ion()
#%%
# %% Learning parameters
n_games = 2000000
epsilon = 0.0
epsilon_step = 0.00075
epsilon_min = 0.01
epsilon_decay = 1/1000
gamma = 0.9
batch_size = 128
buffer_size = 100_000
learning_rate = 0.01
max_episode_step = 200
target_sync_episodes = 1


# %% Neural network, optimizer and loss
in_dim = 4
q_net = torch.nn.Sequential(
    torch.nn.Linear(in_dim, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 4)
)
t_net = copy.deepcopy(q_net)
optimizer = torch.optim.SGD(q_net.parameters(), lr=learning_rate)
loss_function = torch.nn.MSELoss()
#%% buffers
obs_buffer = np.zeros((buffer_size, in_dim))
obs_next_buffer = np.zeros((buffer_size, in_dim))
action_buffer = np.zeros(buffer_size)
reward_buffer = np.zeros(buffer_size)
done_buffer = np.zeros(buffer_size, dtype=np.float32)
# %% Environment
# env = GameController()

#%%
env = GridWorld()
actions = ['left', 'right', 'up', 'down']
action_nums = [0, 1, 2, 3]
# actions = [UP, DOWN, LEFT, RIGHT]
# action_dict = {UP: 0, DOWN:1, LEFT:2, RIGHT:3}
exit_program = False
action_taken = False
clock = pygame.time.Clock()
slow = True
render = True
scores = []
game_tick = 0
learn = False

##
score = 0
episode_step = 0
done = False

observation = env.get_state()
qvals = []

##
scores = []
episode_steps = []
np_array_scores = np.array([])
step_count = 0
print_interval = 10
q_net.load_state_dict(torch.load('1000.pt'))
q_net.eval()
for i in range(n_games):
    np_array_scores = np.append(np_array_scores, int(score))
    env.reset()
    done = False
    score = 0
    game_tick += 1
    frame_tick = 0
    

    while not (done or exit_program):
        if render:
            env.render()
        if slow:
            clock.tick(10)
        frame_tick += 1
        #if score >= 10:
            
            #env.game_over()
    
        # Process game events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit_program = True
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                    pygame.quit()
                    exit_program = True
                if event.key == pygame.K_UP:
                    action, action_taken = 'up', True
                if event.key == pygame.K_DOWN:
                    action, action_taken  = 'down', True
                if event.key == pygame.K_RIGHT:
                    action, action_taken  = 'right', True
                if event.key == pygame.K_LEFT:
                    action, action_taken  = 'left', True
                if event.key == pygame.K_c:
                    
                    env.reset()  
                     
                if event.key == pygame.K_r:
                    render = not render
                if event.key == pygame.K_s:
                    slow = not slow
                if event.key == pygame.K_l:
                    learn = not learn
                if event.key == pygame.K_PLUS:
                    epsilon += 0.001
                if event.key == pygame.K_MINUS:
                    epsilon -= 0.001
                if event.key == pygame.K_COMMA:
                    epsilon += 0.01
                if event.key == pygame.K_PERIOD:
                    epsilon -= 0.01
                if event.key == pygame.K_n:
                    epsilon += 0.1
                if event.key == pygame.K_m:
                    epsilon -= 0.1
               
        # Choose action and step environment
        if np.random.rand() < epsilon:
            action_num = np.random.choice(action_nums)
            
        else:
            action_num = np.argmax(q_net(torch.tensor(observation).float()).detach().numpy())
        observation_next, reward, done = env.step(actions[action_num])
        if frame_tick >= 5000:
            done = True

        score += reward  
        #print(observation, observation_next)
        # with torch.no_grad():
            # qvals.append(q_net(torch.tensor(observation).float())[action_num])

        # Store to buffers
        buffer_index = step_count % buffer_size
        obs_buffer[buffer_index] = observation
        obs_next_buffer[buffer_index] = observation_next
        action_buffer[buffer_index] = action_num
        reward_buffer[buffer_index] = reward
        done_buffer[buffer_index] = done

        # Update to next observation
        observation = observation_next

        # Learn using minibatch from buffer
        if step_count > batch_size and learn:
            # Choose a minibatch            
            batch_idx = np.random.choice(np.minimum(
                buffer_size, step_count), size=batch_size, replace=False)

            # Compute loss function
            out = q_net(torch.tensor(obs_buffer[batch_idx]).float())
            val = out[np.arange(batch_size), action_buffer[batch_idx]]
            with torch.no_grad():
                out_next = t_net(torch.tensor(
                    obs_next_buffer[batch_idx]).float())
                target = torch.tensor(reward_buffer[batch_idx]).float() + \
                    gamma*torch.max(out_next, dim=1).values * \
                    (1-done_buffer[batch_idx])
            loss = loss_function(val, target)

            # Step the optimizer
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Update step counteres
        episode_step += 1
        step_count += 1
        
    #epsilon = epsilon-epsilon_decay
    scores.append(score)
    episode_steps.append(episode_step)

    
    if (i+1) % print_interval == 0:
        # Print average score and number of steps in episode
        # average_score = np.mean(scores[-print_interval:-1])
        # average_episode_steps = np.mean(episode_steps[-print_interval:-1])
        # print(f'Episode={i+1}, Score={average_score:.1f}, Steps={average_episode_steps:.0f}')

        # Plot scores        
        plt.figure(1)
        plt.clf()
        plt.plot(scores, '.')
        plt.plot(scipy.signal.medfilt(scores, 51)[0:-50], linewidth=3)
        plt.title(f'Step {step_count}, eps={epsilon:.3}, learn={learn}, game={game_tick}')
        # plt.ylim(0, 300)
        plt.grid(True)
         
        
        drawnow()
    #print(epsilon)
    if game_tick >=1001:
        print()
        print('Mean score: ', np.mean(np_array_scores[1:]))
        print('variance^2: ', np.var(np_array_scores[1:]))
        np.savetxt('np_array_1000.txt', np_array_scores[1:])
        np.save('np_array_1000.npy', np_array_scores[1:])
        #torch.save(q_net.state_dict(), '1000.pt')
        pygame.quit()
    
                    
    #     # if action_taken:
    #     #     (x, y, gx, gy), reward, done = env.step(action)
    #     #     action_taken = False
        
    #     # Random action
    #     state, reward, done = env.step(np.random.choice(actions))
    #     score += reward
        
    # scores.append(score)
    # # print(f'Score={score}')
    # if game_tick%10 == 0:
    #     plt.figure(1)
    #     plt.clf()
    #     plt.plot(scores, '.')
    #     plt.plot(scipy.signal.medfilt(scores, 51)[0:-50], linewidth=3)
    #     plt.grid(True)
    #     drawnow()

env.close()
