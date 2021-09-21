import pygame
import random
import numpy as np
from pygame import draw
from numpy import array as Vec
from model import DQN
import torch
import copy
import math
import matplotlib.pyplot as plt
import os
import time

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

BoardX = 30 #게임 사이즈
BoardY = 30

PixelPerBlock = 25

screen = pygame.display.set_mode(
    (BoardX*PixelPerBlock, BoardY*PixelPerBlock), pygame.DOUBLEBUF)
clock = pygame.time.Clock()

framerate = 100

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

def norm(vec):
    return abs(vec[0])+abs(vec[1])

def DrawBlock(position, color):
    block = pygame.Rect(position[0]*PixelPerBlock, position[1]
                        * PixelPerBlock, PixelPerBlock, PixelPerBlock)
    pygame.draw.rect(screen, color, block)

def isEqual(vec1, vec2): #두 벡터가 방향이 같은지 확인 
    vec2=vec2/norm(vec2)
    if np.array_equal(vec1,vec2):
        return True
    return False

score_history = []

class Snake:
    def __init__(self): #게임 초기상태
        self.body = [Vec([BoardX//2, BoardY//2]), Vec([BoardX//2, BoardY//2])]
        self.dir = 0
        self.last_dir = 0
        self.consume = False
        self.genApple()
        self.score = 0
        self.last_distance=self.apple-self.body[0]

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
            apple = Vec([temp % BoardX, temp//BoardX])
            coll = False
            if self.isInBody(apple):
                continue
            self.apple = apple
            self.last_distance=self.apple-self.body[0]
            return

    def getState(self): #현재 state 리턴  
        head = self.body[0]
        dir = getDir(self.dir)  # left,foward,right
        distance = [1 for _ in range(3)]
        done = [False, False, False]
        pos = copy.deepcopy([head, head, head])
        valuePerDistance=[0, 0.4, 0.6, 0.8, 0.9]
        for i in range(5):
            for j in range(3):
                if done[j]:
                    continue
                pos[j] += delta[dir[j]]
                if self.isInBody(pos[j]) or self.isOutOfBoard(pos[j]):
                    distance[j]=valuePerDistance[i]
                    done[j] = True
        appledir = [0, 0, 0]
        toapple = self.apple-head
        for i in range(3):
            if np.inner(toapple, delta[dir[i]]) > 0:
                appledir[i] = 1
        if isEqual(dir[1],toapple):
            appledir[1]=1
        else:
            appledir[1]=0
        return torch.FloatTensor(distance+appledir)

    def getReward(self): #사과 먹었으면 10점 아니면 -5점 가까워지면 3
        if self.consume:
            self.consume = False
            return 10
    
        if self.isDead():
            return -100

        now_distance = self.apple - self.body[0]
        if norm(now_distance) < norm(self.last_distance):
            self.last_distance = now_distance
            return 1.5

        return -0.777

    def isOutOfBoard(self, position): #해당 좌표가 보드 밖으로 나갔는지 확인
        if position[0] < 0 or position[0] >= BoardX or position[1] < 0 or position[1] >= BoardY:
            return True
        return False

    def isInBody(self, position): #해당 좌표가 뱀의 몸에 겹치는지 확인
        for elem in self.body:
            if np.array_equal(elem, position):
                return True
        return False

def train(episode):
    os.system("cls")
    Game = Snake() 
    agent = DQN(episode)

    for i in range(episode):
        while True:
            clock.tick(framerate)  #딜레이
            screen.fill(BLACK)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break

            dirs = getDir(Game.dir)
            arr = [Game.getState()]
            action = agent.select_action(arr[0])

            Game.changeDir(dirs[action])
            Game.MoveSnake()
            arr.append(Game.getState())
            reward = Game.getReward()

            agent.memorize(arr[0], action, reward, arr[1])
            agent.optimize_model()
            

            if Game.isDead():
                score_history.append(Game.score)
                agent.decay_epsilon()
                Game.__init__()
                break

            Game.Draw()
            pygame.display.flip()
        print(f"{i+1}/{episode} - 점수 : {score_history[i]} / 최고 점수 : {max(score_history)}")

    plt.plot(score_history)
    plt.title(f"Result of Snake Game in RL that {episode} times learning")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.show()

if __name__ == "__main__":
    train(3000)