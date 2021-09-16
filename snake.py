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

BoardX=15
BoardY=15

PixelPerBlock=50

screen=pygame.display.set_mode((BoardX*PixelPerBlock,BoardY*PixelPerBlock),pygame.DOUBLEBUF)
clock=pygame.time.Clock()

framerate=10
speed=10


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
    def __init__(self):
        self.body=[Vec([BoardX//2,BoardY//2]),Vec([BoardX//2,BoardY//2])]
        self.dir=0
        self.last_dir=0
        self.genApple()
                
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
    
    def genApple(self):
        while True:
            temp=random.randrange(0,BoardX*BoardY)
            apple=Vec([temp%BoardX,temp//BoardY])
            coll=False
            if self.isInBody(apple):
                continue
            self.apple=apple
            return
    
    def getState(self):
        head=self.body[0]
        dir=getDir(self.dir)+[Opposite(self.dir)] #left,foward,right,back
        distance=[1,1,1]
        done=[False,False,False]
        pos=copy.deepcopy([head,head,head])
        for i in range(5):
            for j in range(3):
                if done[j]:
                    continue
                pos[j]+=delta[dir[j]]
                if self.isInBody(pos[j]) or self.isOutOfBoard(pos[j]):
                    done[j]=True
                else:
                    distance[j]-=0.2
        appledir=[0,0,0,0]
        toapple=self.apple-head
        for i in range(4):
            if np.inner(toapple,delta[dir[i]])>0:
                appledir[i]=1
        return distance+appledir
        
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
        print(Game.getState())
        Game.Draw()
        pygame.display.flip()
        
        
MainLoop()
        