# Grid World game

# Import libraries used for this program
 
import pygame
import numpy as np

#%%

class GridWorld():    
    # Rendering?
    rendering = False
    
    # Images
    filenames = ['Pacman.png', 'spøgelse.png', 'door.png', 'death.png']
    images = [pygame.image.load(file) for file in filenames]

    # Colors
    goodColor = (30, 192, 30)
    badColor = (192, 30, 30)
    pathColor = (225, 220, 225)
    wallColor = (157, 143, 130)
    
    def __init__(self, state=None):
        pygame.init()
        self.reward = 0
        if state is None:
            self.x, self.y, self.gx, self.gy, self.board, self.score = self.new_game()            
        else:
            x, y, board, score = state
            self.x, self.y, self.board, self.score = x, y, board.copy(), score
        self.actions = ['left', 'right', 'up', 'down']
        self.done = False

    def get_state(self):
        one_hot_x = np.zeros(10); one_hot_x[self.x] = 1
        one_hot_y = np.zeros(10); one_hot_y[self.y] = 1
        one_hot_gx = np.zeros(10); one_hot_gx[self.gx] = 1
        one_hot_gy = np.zeros(10); one_hot_gy[self.gy] = 1 
        state = (self.x, self.y, self.gx,self.gy)
        return state #np.concatenate([one_hot_x, one_hot_y, one_hot_gx, one_hot_gy])
            
    def step(self, action):
        # Move character
        self.x, self.y, self.board, self.score, self.reward = self.move(self.x, self.y, self.board, self.score, action)

        if self.game_over():
            self.done = True    
            
        # Move ghost
        action = np.random.choice(self.actions)
        self.move_ghost(self.gx, self.gy, self.board, self.score, action)

        if self.game_over():
            self.done = True        
        
        # return observation, reward, done
        return (self.get_state(), self.reward, self.done)
        
    def render(self):
        if not self.rendering:
            self.init_render()
                 
        # Clear the screen
        self.screen.fill((187,173,160))
        
        border = 3
        pygame.draw.rect(self.screen, (187,173,160), pygame.Rect(100,0,600,600))
        for i in range(10):
            for j in range(10):
                val = self.board[i,j]
                col = self.wallColor if val & 8 else self.pathColor
                pygame.draw.rect(self.screen, col, pygame.Rect(100+60*i+border,60*j+border,60-2*border,60-2*border))
                if val>0:
                    x = 105 + 60*i
                    y = 5 + 60*j
                    if val & 1:
                        if self.done:
                            self.screen.blit(self.images[3], (x, y))
                        else:
                            self.screen.blit(self.images[0], (x, y))
                    if val & 2:
                        self.screen.blit(self.images[1], (x, y))

        text = self.scorefont.render("{:}".format(self.score), True, (0,0,0))
        self.screen.blit(text, (790-text.get_width(), 10))
        
        # Draw game over or you won       
        if self.done:
            msg = 'Game over!'
            col = self.badColor
            text = self.bigfont.render(msg, True, col)
            textpos = text.get_rect(centerx=self.background.get_width()/2)
            textpos.top = 300
            self.screen.blit(text, textpos)

        # Display
        pygame.display.flip()

    def reset(self):
        self.x, self.y, self.gx, self.gy, self.board, self.score = self.new_game()

    def close(self):
        pygame.quit()
                 
    def init_render(self):
        self.screen = pygame.display.set_mode([800, 600])
        pygame.display.set_caption('Grid World')
        self.background = pygame.Surface(self.screen.get_size())
        self.rendering = True
        self.clock = pygame.time.Clock()

        # Set up game
        self.bigfont = pygame.font.Font(None, 80)
        self.scorefont = pygame.font.Font(None, 30)
           
    def game_over(self):
        # Are we on a death square?
        if self.board[self.x,self.y] & 8:
            
            return True

        # Are we on the ghost?
        if self.x==self.gx and self.y==self.gy:
            return True
        
        return False
        
        
    def move(self, x, y, board, score, direction='left'):
        newx, newy = x, y
        if direction=='left':
            if x>0:
                if not board[x-1,y] & 8:
                    newx = x-1
        elif direction=='right':
            if x<9:
                if not board[x+1,y] & 8:
                    newx = x+1
        elif direction=='up':
            if y>0:
                if not board[x,y-1] & 8:
                    newy = y-1                
        elif direction=='down':
            if y<9:
                if not board[x,y+1] & 8:
                    newy = y+1
        
        reward = 1
        
        # Update position
        board[x,y] -= 1
        board[newx, newy] += 1
        self.x, self.y = newx, newy
        
                
        # On death?
        if board[newx, newy] & 8:
            reward = -100

        score += reward                        
        return (newx, newy, board, score, reward)
    
    def move_ghost(self, x, y, board, score, direction='left'):
        newx, newy = x, y
        if direction=='left':
            if x>0:
                if not board[x-1,y] & 8:
                    newx = x-1
        elif direction=='right':
            if x<9:
                if not board[x+1,y] & 8:
                    newx = x+1
        elif direction=='up':
            if y>0:
                if not board[x,y-1] & 8:
                    newy = y-1                
        elif direction=='down':
            if y<9:
                if not board[x,y+1] & 8:
                    newy = y+1
        
      
        # Update position
        board[x,y] -= 2
        board[newx, newy] += 2
        self.gx, self.gy = newx, newy
                        
        reward = 0
        score += reward                        
        return (newx, newy, board, score, reward)
       
    def new_game(self):
        board = np.loadtxt('board.txt', dtype=int).T
        if board.shape != (10,10):
            raise Exception('board.txt corrupt')

        start_x, start_y = np.where(board == 0)
        i = np.random.choice(range(len(start_x)), 2, replace=False)
        x, y = start_x[i[0]], start_y[i[0]]
        gx, gy = start_x[i[1]], start_y[i[1]]
        board[x, y] = 1
        board[gx, gy] = 2
        
        self.done = False
        score = 0
        return (x, y, gx, gy, board, score)


