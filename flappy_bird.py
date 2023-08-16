import pygame
from pygame.locals import *
import random
import torch
import torch.nn as nn
import torch.optim as optim
import time

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2) # Two actions: left or right
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.fc(x)

class QLearningAgent:
    def __init__(self, learning_rate=0.00003, discount_factor=0.99, exploration_prob=1.0, exploration_decay=0.9):
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.exploration_decay = exploration_decay

    def select_action(self, state):
        if random.random() < self.exploration_prob:
            return random.randint(0, 1) # Random action
        with torch.no_grad():
            q_values = self.model(torch.FloatTensor(state))
            return torch.argmax(q_values).item() # Greedy action

    def train(self, state, action, reward, next_state, done):
        with torch.no_grad():
            target_q_values = self.model(torch.FloatTensor(next_state))
            max_target_q_value = torch.max(target_q_values)

        q_values = self.model(torch.FloatTensor(state))
        target_q_value = reward + (1 - done) * self.discount_factor * max_target_q_value
        target_q_values = q_values.clone().detach()
        target_q_values[action] = target_q_value

        loss = self.criterion(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if done:
            self.exploration_prob *= self.exploration_decay


pygame.init()

Clock=pygame.time.Clock()
fps = 60


screen_width=550
screen_height=700

screen=pygame.display.set_mode((screen_width,screen_height))
pygame.display.set_caption('flappy bird')


#col and font
font=pygame.font.SysFont('Bauhaus 93',60)
white=(255,255,255)

#define game variable
ground_scroll = 0
scroll_speed = 4
#flying , set to false while starting of the game
flying=False
#add game over variable
game_over=False

#add pipe gap
pipe_gap=180
#fre
pipe_frequency=1500 #millisecond
last_pipe=pygame.time.get_ticks() - pipe_frequency
score=0
pass_pipe=False



#load image
bg=pygame.image.load(r'bg.jpeg')
ground_img=pygame.image.load(r'ground.jpeg')
button_img=pygame.image.load(r'restart.jpeg')

#display score
def draw_text(text,font,text_col,x,y):
    img=font.render(text,True,text_col)
    screen.blit(img,(x,y))


def reset_game():
    #reinitialise the pipes again
    pipe_group.empty()
    #reposition the bird to og position
    flappy.rect.x=100
    flappy.rect.y=int(screen_height/2)
    score=0
    return score




class Bird(pygame.sprite.Sprite):
    def __init__(self,x,y):
        pygame.sprite.Sprite.__init__(self)

        #to make animation we make a list of images and swap
        self.images=[]
        self.index=0
        self.counter=0
        for num in range(1,4):
            img=pygame.image.load(r'bird{}.jpeg'.format(num))
            self.images.append(img)


        #assign the image to sprite
        self.image=self.images[self.index]
        #make a rectangle around it
        self.rect = self.image.get_rect()
        #position the rec
        self.rect.center = [x,y]
        #define vel
        self.vel=0
        #add a triger so mouse hold doesnt work
        self.clicked=False

    def update(self, action):
        #GRAVITY
        if flying==True:#inc vel at each itr
            self.vel+=0.5

            #add a upper limit to vel
            if self.vel>8:
                self.vel=8
            #print(self.vel)
            if self.rect.bottom <768:
                self.rect.y+=int(self.vel)

        if game_over==False:
            #JUMP
            if action==1 and self.clicked==False:
                self.clicked=True
                self.vel=-8
            if action==0:
                self.clicked=False


            #handle the animation
            self.counter +=1
            flap_cooldown=5

            if self.counter>flap_cooldown:
                #reset the counter
                self.counter=0
                self.index+=1
                if self.index >= len(self.images):
                    self.index=0
            self.image = self.images[self.index]

            #rotate th bird while falling
            self.image=pygame.transform.rotate(self.images[self.index],self.vel*-2)
        else:
            self.image=pygame.transform.rotate(self.images[self.index],-90)

class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position, bottom):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(r'pipe.jpeg')
        self.rect = self.image.get_rect()
        self.opening_y = y  # Add this line to set the opening y-coordinate
        self.bottom = bottom

        if position == 1:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = [x, y - int(pipe_gap / 2)]
            self.opening_y = y - int(pipe_gap / 2)
        if position == -1:
            self.rect.topleft = [x, y + int(pipe_gap / 2)]
            self.opening_y = y + int(pipe_gap / 2)


    def update(self):
        self.rect.x -= scroll_speed
        if self.rect.right<0:
            self.kill()

class Button():
    def __init__(self,x,y,image):
        self.image=image
        self.rect=self.image.get_rect()
        self.rect.topleft=(x,y)

    def draw(self):

        #define the action
        action=False

        #get mouse posn
        pos=pygame.mouse.get_pos()
        #check if mouse is over the button
        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0]==1:
                action=True

        #draw the button
        screen.blit(self.image,(self.rect.x,self.rect.y))

        return action



bird_group = pygame.sprite.Group()
pipe_group=pygame.sprite.Group()

flappy = Bird(100,int(screen_height/2))

bird_group.add(flappy)


#create restart button instance
button=Button(screen_width//2 -50, screen_height //2 -100,button_img)


def get_state_vector(bird_group, pipe_group):
    bird = bird_group.sprites()[0]

    if len(pipe_group) > 0:
        closest_pipe = None
        min_distance = float('inf')
        height = 0
        bd = bird.rect.centery

        for pipe in pipe_group.sprites():
            if not pipe.bottom:
                continue

            distance = pipe.rect.right - bird.rect.left

            if distance < min_distance and distance > 0:
                min_distance = distance
                closest_pipe = pipe
                height = abs(bird.rect.centery - closest_pipe.opening_y)

        if closest_pipe:
            is_space_higher = bird.rect.centery < closest_pipe.opening_y
            distance_from_pipe = min_distance

            if is_space_higher:
                return [0, 1, closest_pipe.opening_y / 700, bd/700]
            else:
                return [1, 0, closest_pipe.opening_y / 700, bd/700]

    # If no pipe is present, return a default state

    return [random.random(), random.random(), random.random(), random.random()]

agent = QLearningAgent()

run=True
curr_score = 0
inc = False

while run:

    reward = 0
    Clock.tick(fps)

    #to load the bg image on screen we use blit
    #draw bg
    screen.blit(bg,(0,0))

    bird_group.draw(screen)

    state_vector = get_state_vector(bird_group, pipe_group)
    action = agent.select_action(state_vector)

    bird_group.update(action)

    #similarly add update for pipr
    pipe_group.draw(screen)

    #draw and scroll the ground
    screen.blit(ground_img,(ground_scroll,768))

    #check score
    if len(pipe_group)>0:
        if bird_group.sprites()[0].rect.left>pipe_group.sprites()[0].rect.left\
            and bird_group.sprites()[0].rect.right<pipe_group.sprites()[0].rect.right\
            and pass_pipe == False:
            pass_pipe=True
        if pass_pipe==True:
            if bird_group.sprites()[0].rect.left>pipe_group.sprites()[0].rect.left and game_over == False:
                score +=1
                pass_pipe=False

    draw_text(str(score // 2),font,white,int(screen_width/2),20)
    new_score = score // 2
    if new_score > curr_score:
        curr_score = new_score
        inc = True
        reward = 50

    else:
        inc = False


    ##look for collison
    if pygame.sprite.groupcollide(bird_group,pipe_group,False,False) or flappy.rect.top<0:
        game_over=True
        reward = -10
    #check if bird has hit grounf
    #check if rect as gone beyond ground
    if flappy.rect.bottom >=768:
        game_over=True
        reward = -10
        flying=False



    if game_over==False and flying==True:

        #create extra pipes when game is running
        time_now=pygame.time.get_ticks()
        if time_now-last_pipe>pipe_frequency:
                #that means we can add extra pipe
            pipe_height=random.randint(-100,100)
            btm_pipe=Pipe(screen_width,int(screen_height/2)+pipe_height,-1, True)
            top_pipe=Pipe(screen_width,int(screen_height/2)+pipe_height,1, False)
            pipe_group.add(btm_pipe)
            pipe_group.add(top_pipe)
            last_pipe=time_now

        #it increases on the left side
        ground_scroll -= scroll_speed
        if abs(ground_scroll)>35:
            ground_scroll=0

        pipe_group.update()

    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            run=False
        if flying==False and game_over==False:
            flying=True

    next_state_vector = get_state_vector(bird_group, pipe_group)
    #check for gameover and reset
    if game_over==True:
        agent.train(state_vector, action, reward, next_state_vector, True)
        game_over=False
        time.sleep(0.7)
        score=reset_game()
        continue

    agent.train(state_vector, action, reward, next_state_vector, False)
    pygame.display.update()

pygame.quit()
