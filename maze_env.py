"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example.
The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import time
import random as rd
import sys
# import matplotlib.pyplot as plt

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

'''考虑将像素换成20'''
UNIT = 20  # pixels
MAZE_H = 40  # grid height
MAZE_W = 40  # grid width

ls = np.arange(0, 40)
random_number = rd.sample(range(1, 40), 20)
random_number_2 = rd.sample(range(1, 40), 20)

'''这里可以加一个标题label'''


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2              # 加大n_features2——>20，需要改变状态的输出值
        self.title('Analog map')
        self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        # 设置背景幕布
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAZE_H * UNIT,
                                width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):  # 画每一列,中间间隔20个像素点
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)  # canvas.create_line(x1,y1,x2,y2,width,fill,dash)
        for r in range(0, MAZE_H * UNIT, UNIT):  # 画每一行，中间间隔20个像素点
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # 设置黑洞
        self.hell1 = np.array(range(0, 20))
        for i in range(0, 20):
            arr = np.array([ls[random_number[i]] * UNIT, ls[random_number_2[i]] * UNIT])
            # canvas.create_rectangle()会返回一个对象ID，就是调试是看到的81、82这些整数
            self.hell1[i] = self.canvas.create_rectangle(arr[0] + 1, arr[1] + 1, arr[0] + 19, arr[1] + 19, fill='black')
        # print([self.canvas.coords(self.hell1[i]) for i in range(0,20)])  # 这里必须写列表理解

        # create oval
        oval_center = origin + UNIT * 30
        self.oval = self.canvas.create_oval(
            oval_center[0] + 1, oval_center[1] + 1,
            oval_center[0] + 19, oval_center[1] + 19,
            fill='yellow')
        self.oval_position=self.canvas.coords(self.oval)

        # 设置起始位置
        self.rect = self.canvas.create_rectangle(
            origin[0] + 1, origin[1] + 1,
            origin[1] + 19, origin[1] + 19,
            fill='red'
        )  # 起点位于左上角
        self.start=self.canvas.coords(self.rect)

        self.canvas.pack()

    def reset(self):
        self.update()  # 更新视图
        time.sleep(0.1)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] + 1, origin[1] + 1,
            origin[0] + 19, origin[1] + 19,
            fill='red')
        self.value = (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (
                MAZE_H * UNIT)
        # print(self.value)
        # return observation
        return (np.array(self.canvas.coords(self.rect)[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (
                MAZE_H * UNIT)

    def step(self, action):
        s = self.canvas.coords(self.rect)  # coords(*args) 返回画布对象的坐标(x1,y1,x2,y2) -----为什么这里的s是1*2的数组，而不是1*4
        base_action = np.array([0, 0])
        if action == 0:  # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:  # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:  # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:  # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        current_coords = self.canvas.coords(self.rect)
        old_distance = np.square(current_coords[0]-self.oval_position[0])+np.square(current_coords[1]-self.oval_position[1])
        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent
        # 移动红色方块，self.move(canvas_object,x,y) 将canvas_object移动到(x,y)的位置，x是距左上角的水平距离，y是距左上角的垂直距离

        next_coords = self.canvas.coords(self.rect)  # next state
        # 计算下一状态与终点的距离
        distance = np.square(next_coords[0]-self.oval_position[0])+np.square(next_coords[1]-self.oval_position[1])

        # reward function
        if next_coords == self.canvas.coords(self.oval):
            reward = 200
            done = True
        elif next_coords in [self.canvas.coords(self.hell1[i]) for i in range(0, 20)]:
            reward = -50
            done = True
        # print([self.canvas.coords(self.hell1[i]) for i in range(0,20)])
        elif distance > old_distance:  # 添加距离的奖励，当下一步距终点的距离比上一步大时，给一个更大的负收益
            reward = -20
            done = False
        elif distance <= old_distance:  # 当下一步距终点的距离小于等于上一步时，给一个较小的负收益
            reward = -5
            done = False
        else:  # ————————————怎么让红点不要在一个地方转圈呢————————————————————
            reward = -1
            done = False
            # if reward>=-60:
            #  done = False
            # else:
            #   done = True
        s_ = (np.array(next_coords[:2]) - np.array(self.canvas.coords(self.oval)[:2])) / (MAZE_H * UNIT)
        # print(s_)
        return s_, reward, done

    def render(self):
        # time.sleep(0.01)
        self.update()




    '''        # 边框所在坐标的集合
            # cheek_gather = self.canvas.coords()
            self.cheek_collection = np.array(range(0, 156))
            for i in range(0, 156):
                if self.canvas.coords(self.rect)[0] == 1:
                    self.cheek_collection[i] = self.canvas.create_rectangle(self.canvas.coords(self.rect))
                elif self.canvas.coords(self.rect)[0] == 781:
                    self.cheek_collection[i] = self.canvas.create_rectangle(self.canvas.coords(self.rect))
                elif self.canvas.coords(self.rect)[1] == 21:
                    self.cheek_collection[i] = self.canvas.create_rectangle(self.canvas.coords(self.rect))
                elif self.canvas.coords(self.rect)[1] == 801:
                    self.cheek_collection[i] = self.canvas.create_rectangle(self.canvas.coords(self.rect))
    '''
    '''        elif next_coords in [self.canvas.coords(self.cheek_collection[i]) for i in range(0,156)]:
                reward = -10
                done = True
    '''
