import pygame
import random
import numpy as np
from pygame import draw
from numpy import array as Vec

import copy

WHITE=(255,255,255)
BLACK=(0,0,0)
RED=(255,0,0)
GREEN=(0,255,0)

BoardX = 30 #게임 사이즈
BoardY = 30

PixelPerBlock = 30

screen = pygame.display.set_mode(
    (BoardX*PixelPerBlock, BoardY*PixelPerBlock), pygame.DOUBLEBUF)

pygame.display.set_caption("Snake Game in RL")

clock = pygame.time.Clock()

framerate = 30


delta=[
    Vec([1,0]), #Right
    Vec([0,1]), #Down
    Vec([-1,0]),#Left
    Vec([0,-1]) #Up
]

KEY2DIR={
    pygame.K_UP:3,
    pygame.K_DOWN:1,
    pygame.K_LEFT:2,
    pygame.K_RIGHT:0
}

def Opposite(dir): 
    return (dir+2)%4

def getDir(dir):
    foward=dir
    right=(dir+1)%4
    left=(dir-1)%4
    return [left,foward,right]

def DrawBlock(position,color):
    block=pygame.Rect(position[0]*PixelPerBlock,position[1]*PixelPerBlock,PixelPerBlock,PixelPerBlock)
    pygame.draw.rect(screen,color,block)

class Snake:
    def __init__(self): #게임 초기상태
        self.board=np.zeros((BoardX,BoardY))
        self.body = [Vec([BoardX//2, BoardY//2]), Vec([BoardX//2, BoardY//2])]
        self.board[BoardX//2][BoardY//2]=2
        self.dir = 0
        self.last_dir = 0
        self.consume = False
        self.genApple()
        self.score = 0

    def Draw(self):
        for position in self.body:
            DrawBlock(position,WHITE)
        DrawBlock(self.apple,RED)

    def MoveSnake(self):
        head=self.body[0]+delta[self.dir]
        self.last_dir=self.dir
        self.body.insert(0,head)
        if np.array_equal(head,self.apple):
            self.genApple()
        else:
            self.body.pop()

    def isDead(self):
        head=self.body[0]
        if head[0]<0 or head[0]>=BoardX or head[1]<0 or head[1]>=BoardY:
            return True
        for i in range(1,len(self.body)):
            if np.array_equal(self.body[0],self.body[i]):
                return True
        return False

    def changeDir(self,dir_NEW):
        if Opposite(self.last_dir) == dir_NEW or self.last_dir == dir_NEW:
            return False
        self.dir=dir_NEW
        return True

    def genApple(self): #사과 생성
        while True:
            temp = random.randrange(0, BoardX*BoardY)
            apple = Vec([temp % BoardX, temp//BoardX])
            if apple[0]==0 or apple[0]==BoardX-1 or apple[1]==0 or apple[1]==BoardY-1:
                continue
            if self.board[apple[0]][apple[1]]>0:
                continue
            self.apple = apple
            self.last_distance=self.apple-self.body[0]
            return

    def isOutOfBoard(self,position):
        if position[0]<0 or position[0]>=BoardX or position[1]<0 or position[1]>=BoardY:
            return True
        return False

    def isInBody(self,position):
        for elem in self.body:
            if np.array_equal(elem,position):
                return True
        return False



def MainLoop():
    Game=Snake()
    running=True
    last_input=[]
    while running:
        clock.tick(framerate)
        screen.fill(BLACK)
        for dir in last_input:
            if Game.changeDir(dir):
                break
        last_input.clear()
        gotInput=False
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                running=False
            if event.type==pygame.KEYDOWN:
                if event.key in KEY2DIR:
                    if gotInput:
                        last_input.append(KEY2DIR[event.key])
                    else:
                        temp=Game.changeDir(KEY2DIR[event.key])
                        gotInput=True or temp is True
        Game.MoveSnake()
        if Game.isDead():
            Game.__init__()
        Game.Draw()
        pygame.display.flip()


MainLoop()