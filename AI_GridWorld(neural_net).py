# Grid World: AI-controlled play

# Instructions:
#   Move up, down, left, or right to move the character. The 
#   objective is to find the key and get to the door
#
# Control:
#    arrows  : Merge up, down, left, or right
#    s       : Toggle slow play
#    a       : Toggle AI player
#    d       : Toggle rendering 
#    r       : Restart game
#    q / ESC : Quit

from GridWorld import GridWorld
import numpy as np
import pygame
from collections import defaultdict 
import cv2
import matplotlib.pyplot as plt
import torch
import copy

# Initialize the environment
env = GridWorld()
env.reset()
x, y, has_key = env.get_state()

# Definitions and default settings
actions = ['left', 'right', 'up', 'down']
exit_program = False
action_taken = False
slow = False
runai = True
render = True
optimize = True
done = False

# Game clock
clock = pygame.time.Clock()

# INSERT YOUR CODE HERE (1/2)
# Define data structure for q-table
# Here, we will use optimistic initialization and assume all state-actions 
#   have quality 0. This is optimistic, because each step yields reward -1 and 
#   only the key and door give positive rewards (50 and 100)
# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
D_in, H, D_out = 40*30, 100, 4

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')
# list(model.parameters())[2].data*=0.0001
# list(model.parameters())[3].data -= 10

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Total length of frame buffer
F = 512

# Mini batch size
B = 512

# Update interval
G = 512

frame_buffer_0 = torch.zeros((F, D_in))
frame_buffer_1 = torch.zeros((F, D_in))
action_buffer = torch.zeros(F, dtype=torch.long)
target_buffer = torch.zeros(F)
reward_buffer = torch.zeros(F)
done_buffer = torch.zeros(F, dtype=torch.bool)

loss_iter = []

gamma = 1
epsilon = 1


buf = torch.zeros((4, D_in))
env.render()
img_0 = cv2.resize(pygame.surfarray.array3d(env.screen)/256, (30,40)).mean(2).T 
x_0 = torch.tensor(img_0.ravel(), dtype=torch.float)
buf[0] = x_0
env.step('right')
env.render()
img_0 = cv2.resize(pygame.surfarray.array3d(env.screen)/256, (30,40)).mean(2).T 
x_0 = torch.tensor(img_0.ravel(), dtype=torch.float)
buf[1] = x_0
env.step('left')
env.render()
img_0 = cv2.resize(pygame.surfarray.array3d(env.screen)/256, (30,40)).mean(2).T 
x_0 = torch.tensor(img_0.ravel(), dtype=torch.float)
buf[2] = x_0
env.step('left')
env.render()
img_0 = cv2.resize(pygame.surfarray.array3d(env.screen)/256, (30,40)).mean(2).T 
x_0 = torch.tensor(img_0.ravel(), dtype=torch.float)
buf[3] = x_0



env.reset()
x, y, has_key = env.get_state()

# END OF YOUR CODE (1/2)

t = 0

while not exit_program:

    if render:
        env.render()
    
    # Slow down rendering to 5 fps
    if slow and runai:
        clock.tick(5)       

    # Automatic reset environment in AI mode
    if done and runai:
        env.reset()
        x, y, has_key = env.get_state()
        done = False
        continue       
               
    # Process game events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit_program = True
        if event.type == pygame.KEYDOWN:
            if event.key in [pygame.K_ESCAPE, pygame.K_q]:
                exit_program = True
            if event.key == pygame.K_UP:
                action, action_taken = 'up', True
            if event.key == pygame.K_DOWN:
                action, action_taken  = 'down', True
            if event.key == pygame.K_RIGHT:
                action, action_taken  = 'right', True
            if event.key == pygame.K_LEFT:
                action, action_taken  = 'left', True
            if event.key == pygame.K_r:
                env.reset()   
                
            if event.key == pygame.K_d:
                render = not render
            if event.key == pygame.K_s:
                slow = not slow
            if event.key == pygame.K_o:
                optimize = not optimize
                print('Optimize: {}'.format(optimize))
            if event.key == pygame.K_a:
                runai = not runai
                clock.tick(5)
    
    # AI controller (enable/disable by pressing 'a')
    if runai or action_taken:
        # INSERT YOUR CODE HERE (2/2)

        # if t<F:
        #     img_0 = cv2.resize(pygame.surfarray.array3d(env.screen)/256, (30,40)).mean(2).T 
        #     x_0 = torch.tensor(img_0.ravel(), dtype=torch.float)
        #     y_0 = model(x_0)
            
        #     if (np.random.rand()>epsilon) or (not optimize):
        #         action_num_0 = torch.argmax(y_0.detach())
        #     else:
        #         action_num_0 = np.random.randint(4)
        #     action = actions[action_num_0]        
            
    
        #     # # 2. step the environment
        #     (x, y, has_key), reward, done = env.step(action)
        #     env.render()
    
        #     img_1 = cv2.resize(pygame.surfarray.array3d(env.screen)/256, (30,40)).mean(2).T 
        #     x_1 = torch.tensor(img_1.ravel(), dtype=torch.float)
        #     y_1 = model(x_1)
    
        #     frame_buffer_0[t%F] = x_0
        #     frame_buffer_1[t%F] = x_1
        #     action_buffer[t%F] = action_num_0
        #     target_buffer[t%F] = reward if done else reward+gamma*torch.max(y_1.detach())
        #     reward_buffer[t%F] = reward
        #     done_buffer[t%F] = done
        # else:
        #     y = model(frame_buffer_1).detach()
        #     target_buffer = reward_buffer + gamma*y.max(dim=1).values * done_buffer
        
        # if (t%G == G-1) and (t>B) :
        #     for n in range(100):
        #         max_idx = t if t<F else F
        #         batch_idx = np.random.choice(range(max_idx), B, replace=False)
        #         y_buffer = model(frame_buffer_0[batch_idx])
        #         y_pred = y_buffer.gather(1, action_buffer[batch_idx,None])
        #         loss = loss_fn(y_pred.squeeze(), target_buffer[batch_idx])
        #         loss_iter.append(loss.item())
        #         # print(loss.item())
    
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()

        #     print(model(buf).detach())

        #     plt.subplot(221)
        #     plt.plot(model(frame_buffer_0).detach(), '.', markersize=1)
        #     plt.subplot(222)
        #     plt.plot(loss_iter, '.', markersize=1)
        #     plt.show()
        
        frame_buffer_0 = buf[0:3]
        frame_buffer_1 = buf[1:4]        
        action_buffer = torch.tensor([1,0,0])
        reward_buffer = torch.tensor([50,-1,100])
        y_1 = model(frame_buffer_1).detach()
        target_buffer = reward_buffer + gamma*y_1.gather(1, torch.tensor([0, 0, 0])[:,None]).squeeze()
        target_buffer[2] = reward_buffer[2]
        
        for n in range(100):
            y_buffer = model(frame_buffer_0)
            y_pred = y_buffer.gather(1, action_buffer[:,None])
            loss = loss_fn(y_pred.squeeze(), target_buffer)
            loss_iter.append(loss.item())
            # print(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(model(frame_buffer_0).detach())


            
        
        # if reward>0:
        #     print('Reward: {}'.format(reward))

        # # 3. update q table
        # if optimize:
        #     env.render()
        #     img_1 = cv2.resize(pygame.surfarray.array3d(env.screen)/256, (30,40)).mean(2).T 
        #     with torch.no_grad():
        #         x_1 = torch.tensor(img_1.ravel(), dtype=torch.float)
        #         x_1 -= x_1.mean()
        #         y_1 = model_fixed(x_1)*o_scale+o_offset
        #         action_num_1 =  torch.argmax(y_1)
        #         if done:
        #             target = torch.tensor(reward, dtype=torch.float)
        #         else:
        #             target = reward+gamma*y_1[action_num_1]
           
        #     loss = loss_fn(y_0[action_num_0], target)   
        #     # print(loss.item())
        #     optimizer.zero_grad()
        #     loss.backward()
        #     # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        #     optimizer.step()
        
        action_taken = False
 

        # END OF YOUR CODE (2/2)
    
    # Human controller        
    else:
        if action_taken:
            (x, y, has_key), reward, done = env.step(action)
            action_taken = False

    t += 1

env.close()
