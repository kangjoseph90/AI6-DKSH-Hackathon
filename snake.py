import pygame
import random
import numpy as np
from pygame import draw
from numpy import array as Vec
from model import DQN
import torch
import copy

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

BoardX = 15 #게임 사이즈
BoardY = 15

PixelPerBlock = 50

screen = pygame.display.set_mode(
    (BoardX*PixelPerBlock, BoardY*PixelPerBlock), pygame.DOUBLEBUF)
clock = pygame.time.Clock()

framerate = 10
speed = 10


delta = [ #방향당 위치 변화값
    Vec([1, 0]),  # Right
    Vec([0, 1]),  # Down
    Vec([-1, 0]),  # Left
    Vec([0, -1])  # Up
]

KEY2DIR = { 
    pygame.K_UP: 3,
    pygame.K_DOWN: 1,
    pygame.K_LEFT: 2,
    pygame.K_RIGHT: 0
}


def Opposite(dir): #정면방향 기준 뒤쪽 방향 (절대방향) 리턴
    return (dir+2) % 4


def getDir(dir): #정면방향 기준 [좌측, 정면, 우측] (절대방향) 리턴
    foward = dir
    right = (dir+1) % 4
    left = (dir-1) % 4
    return [left, foward, right]


def DrawBlock(position, color):
    block = pygame.Rect(position[0]*PixelPerBlock, position[1]
                        * PixelPerBlock, PixelPerBlock, PixelPerBlock)
    pygame.draw.rect(screen, color, block)


class Snake:
    def __init__(self): #게임 초기상태
        self.body = [Vec([BoardX//2, BoardY//2]), Vec([BoardX//2, BoardY//2])]
        self.dir = 0
        self.last_dir = 0
        self.genApple()

    def Draw(self): #화면 출력
        for position in self.body:
            DrawBlock(position, WHITE)
        DrawBlock(self.apple, RED)

    def MoveSnake(self): #현재 dir로 이동
        head = self.body[0]+delta[self.dir]
        self.last_dir = self.dir
        self.body.insert(0, head)
        if np.array_equal(head, self.apple):
            self.genApple()
        else:
            self.body.pop()

    def isDead(self): #죽었는지 확인
        head = self.body[0]
        if head[0] < 0 or head[0] >= BoardX or head[1] < 0 or head[1] >= BoardY:
            return True
        for i in range(1, len(self.body)):
            if np.array_equal(self.body[0], self.body[i]):
                return True
        return False

    def changeDir(self, dir_NEW): #절대방향으로 방향 변경
        if Opposite(self.last_dir) == dir_NEW or self.last_dir == dir_NEW:
            return False
        self.dir = dir_NEW
        return True

    def genApple(self): #사과 생성
        while True:
            temp = random.randrange(0, BoardX*BoardY)
            apple = Vec([temp % BoardX, temp//BoardY])
            coll = False
            if self.isInBody(apple):
                continue
            self.apple = apple
            return

    def getState(self): #현재 state 리턴  
        head = self.body[0]
        dir = getDir(self.dir)+[Opposite(self.dir)]  # left,foward,right,back
        distance = [1, 1, 1]
        done = [False, False, False]
        pos = copy.deepcopy([head, head, head])
        for i in range(5):
            for j in range(3):
                if done[j]:
                    continue
                pos[j] += delta[dir[j]]
                if self.isInBody(pos[j]) or self.isOutOfBoard(pos[j]):
                    done[j] = True
                else:
                    distance[j] -= 0.2
        appledir = [0, 0, 0, 0]
        toapple = self.apple-head
        for i in range(4):
            if np.inner(toapple, delta[dir[i]]) > 0:
                appledir[i] = 1
        return torch.FloatTensor(distance+appledir)

    def isOutOfBoard(self, position): #해당 좌표가 보드 밖으로 나갔는지 확인
        if position[0] < 0 or position[0] >= BoardX or position[1] < 0 or position[1] >= BoardY:
            return True
        return False

    def isInBody(self, position): #해당 좌표가 뱀의 몸에 겹치는지 확인
        for elem in self.body:
            if np.array_equal(elem, position):
                return True
        return False

def MainLoop(episode=50):
    Game = Snake() 
    agent = DQN()
    
    running = True
    while running:
        for i in range(episode):
            clock.tick(framerate)  #딜레이
            screen.fill(BLACK)
            for event in pygame.event.get(): #종료시 loop break
                if event.type == pygame.QUIT:
                    running = False
            
            dirs = getDir(Game.dir)
            try: Game.changeDir(dirs[agent.select_action(Game.getState())])
            except: Game.changeDir(dirs[1])
            Game.MoveSnake() #현재 방향으로 이동

            # if Game.

            agent.memorize(Game.getState(), agent.select_action(Game.getState()), reward, next_state)

            if Game.isDead(): #뒤지면 초기화
                Game.__init__()
            Game.Draw()
            pygame.display.flip()


MainLoop()
