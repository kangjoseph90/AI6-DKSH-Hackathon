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

framerate = 30

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

score_history = []

class Snake:
    def __init__(self): #게임 초기상태
        self.body = [Vec([BoardX//2, BoardY//2]), Vec([BoardX//2, BoardY//2])]
        self.dir = 0
        self.last_dir = 0
        self.consume = False
        self.genApple()
        self.score = 0

    def Draw(self): #화면 출력
        for position in self.body:
            DrawBlock(position, WHITE)
        DrawBlock(self.apple, RED)

    def MoveSnake(self): #현재 dir로 이동
        head = self.body[0]+delta[self.dir]
        self.last_dir = self.dir
        self.body.insert(0, head)
        if np.array_equal(head, self.apple):
            self.consume = True
            self.score += 1
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

    def getReward(self): #사과 먹었으면 10점 아니면 0점
        if self.consume:
            self.consume=False
            return 10
        return -1

    def isOutOfBoard(self, position): #해당 좌표가 보드 밖으로 나갔는지 확인
        if position[0] < 0 or position[0] >= BoardX or position[1] < 0 or position[1] >= BoardY:
            return True
        return False

    def isInBody(self, position): #해당 좌표가 뱀의 몸에 겹치는지 확인
        for elem in self.body:
            if np.array_equal(elem, position):
                return True
        return False

def MainLoop(episode=100):
    Game = Snake() 
    agent = DQN()

    for i in range(episode):
        while True:
            clock.tick(framerate)  #딜레이
            screen.fill(BLACK)
            
            dirs = getDir(Game.dir)
            arr = [Game.getState()]
            reward = Game.getReward()
            action = agent.select_action(Game.getState())

            Game.changeDir(dirs[action])
            Game.MoveSnake()
            arr.append(Game.getState())

            agent.memorize(arr[0], action, reward, arr[1])
            agent.optimize_model()

            if Game.isDead():
                score_history.append(Game.score)
                Game.__init__()
                break

            Game.Draw()
            pygame.display.flip()
        print(f"{i+1}번째 에피소드 - 점수 : {score_history[i]}")


MainLoop()