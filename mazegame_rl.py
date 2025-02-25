import pygame
import numpy as np
import random
from collections import defaultdict
import sys
import matplotlib.pyplot as plt
# 环境配置
GRID_SIZE = 8
CELL_SIZE = 75
WIDTH, HEIGHT = GRID_SIZE*CELL_SIZE, GRID_SIZE*CELL_SIZE

# 颜色配置
COLORS = {
    'background': (240, 240, 240),
    'wall': (48, 48, 48),
    'agent': (255, 87, 51),
    'goal': (76, 175, 80),
    'trap': (255, 193, 7),
    'moving_trap': (244, 67, 54)
}

# 强化学习参数
class RLConfig:
    LEARNING_RATE = 0.7
    DISCOUNT_FACTOR = 0.99
    EPISODES = 2500
    INIT_EPSILON = 1.0
    MIN_EPSILON = 0.01
    EPSILON_DECAY = 0.997
    MAX_STEPS = 50

class MazeGame:
    def __init__(self, render=False):
        self.render_mode = render
        # 增加更多墙壁和陷阱
        self.walls = [
            (0,4),(1,0),(1,1),(1,3),(1,4),(1,6),
            (2,3),(2,7),(2,8),(3,1),(3,7),
            (4,1),(4,4),(5,1),(5,2),(5,3),(5,4),(5,5),
            (6,6),(7,1),(7,4)
        ]
        self.traps = [
            (3,5),(0,3),(5,6) 
        ]
        # 增加移动陷阱，并确保它们不会完全封死路径
        self.moving_traps = [
            {'pos': (6, 3), 'dir': 0},  # 垂直移动
            {'pos': (5, 6), 'dir': 1}   # 水平移动
        ]
        self.start = (0, 0)
        self.goal = (GRID_SIZE-1, GRID_SIZE-1)
        
        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("QLearning迷宫")


    def reset(self):
        self.agent_pos = list(self.start)
        return tuple(self.agent_pos)

    def get_valid_actions(self, pos):
        row, col = pos
        actions = []
        if row > 0 and (row-1, col) not in self.walls: actions.append(0)  # 上
        if row < GRID_SIZE-1 and (row+1, col) not in self.walls: actions.append(1)  # 下
        if col > 0 and (row, col-1) not in self.walls: actions.append(2)  # 左 
        if col < GRID_SIZE-1 and (row, col+1) not in self.walls: actions.append(3)  # 右
        return actions

    def move_traps(self):
        for trap in self.moving_traps:
            row, col = trap['pos']
        
        # 垂直移动：沿y轴方向
            if trap['dir'] == 0:  # 垂直方向移动
                new_row = row + 1 if row < GRID_SIZE - 1 else row - 1
                if new_row != row:  # 确保陷阱的位置变化了
                # 确保新位置不与墙壁或其他陷阱重叠
                   if (new_row, col) not in self.walls and (new_row, col) not in [t['pos'] for t in self.moving_traps]:
                        trap['pos'] = (new_row, col)
                else:
                       continue  # 如果新位置无效，则跳过此步
            
            # 如果到达边界，改变方向
                if new_row in [0, GRID_SIZE - 1]:
                    trap['dir'] = 1  # 改为水平方向移动
        
        # 水平移动：沿x轴方向
            elif trap['dir'] == 1:  # 水平方向移动
                new_col = col + 1 if col < GRID_SIZE - 2 else col - 1
                if new_col != col:  # 确保陷阱的位置变化了
                # 确保新位置不与墙壁或其他陷阱重叠
                    if (row, new_col) not in self.walls and (row, new_col) not in [t['pos'] for t in self.moving_traps]:
                        trap['pos'] = (row, new_col)
                    else:
                        continue  # 如果新位置无效，则跳过此步
            
            # 如果到达边界，改变方向
                if new_col in [0, GRID_SIZE - 1]:
                    trap['dir'] = 0  # 改为垂直方向移动


    def step(self, action):
        self.move_traps()
        new_pos = list(self.agent_pos)
    
    # 执行动作
        if action == 0: new_pos[0] -= 1  # 上
        elif action == 1: new_pos[0] += 1  # 下
        elif action == 2: new_pos[1] -= 1  # 左
        elif action == 3: new_pos[1] += 1  # 右
    
    # 碰撞检测
        new_pos = self._check_collision(tuple(new_pos))
        self.agent_pos = new_pos
    
        done = False
        reward = -0.2  # 基础惩罚
        dist = abs(new_pos[0]-self.goal[0]) + abs(new_pos[1]-self.goal[1])
    
    # 奖励判断
        if tuple(new_pos) == self.goal:
            reward = 100
            done = True
        elif tuple(new_pos) in self.traps:
            reward = -25  # 遇到固定陷阱，扣分
            done = True
        elif any(new_pos == trap['pos'] for trap in self.moving_traps):  # 检查是否与移动陷阱碰撞
            reward = -25  # 遇到移动陷阱，扣分
            done = True
    
        if self.render_mode:
            self.render()
        
        return tuple(new_pos), reward, done


    def _check_collision(self, pos):
        if pos in self.walls:
            return list(self.agent_pos)
        return list(np.clip(pos, 0, GRID_SIZE-1))

    def render(self):
        self.screen.fill(COLORS['background'])
        
        # 绘制网格
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                rect = pygame.Rect(j*CELL_SIZE, i*CELL_SIZE, CELL_SIZE-2, CELL_SIZE-2)
                if (i,j) in self.walls:
                    pygame.draw.rect(self.screen, COLORS['wall'], rect)
                elif (i,j) == self.goal:
                    pygame.draw.rect(self.screen, COLORS['goal'], rect)
                elif (i,j) in self.traps:
                    pygame.draw.rect(self.screen, COLORS['trap'], rect)
        
        # 移动陷阱
        for trap in self.moving_traps:
            row, col = trap['pos']
            rect = pygame.Rect(col*CELL_SIZE, row*CELL_SIZE, CELL_SIZE-2, CELL_SIZE-2)
            pygame.draw.rect(self.screen, COLORS['moving_trap'], rect)
        
        # 绘制智能体
        row, col = self.agent_pos
        agent_rect = pygame.Rect(col*CELL_SIZE+15, row*CELL_SIZE+15, CELL_SIZE-30, CELL_SIZE-30)
        pygame.draw.ellipse(self.screen, COLORS['agent'], agent_rect)
        
        pygame.display.flip()

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: [0.0]*4)  # 乐观初始化
    
    def choose_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.choice([0,1,2,3])
        return np.argmax(self.q_table[state])
    
    def update_q(self, state, action, reward, next_state):
        old_value = self.q_table[state][action]
        max_future = np.max(self.q_table[next_state])
        new_value = (1 - RLConfig.LEARNING_RATE) * old_value + RLConfig.LEARNING_RATE * (reward + RLConfig.DISCOUNT_FACTOR * max_future)
        self.q_table[state][action] = new_value
def plot_training_metrics(success_history, step_history, reward_history):
    episodes = np.arange(len(success_history))

    window_size = 100  # 平滑窗口大小
    valid_range = len(success_history) - window_size + 1
    smooth_x = episodes[:valid_range]  # 使 x 轴长度匹配平滑后的 y 轴

    # 绘制成功率变化
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(smooth_x, np.convolve(success_history, np.ones(window_size)/window_size, mode="valid"), label="success rate (100 episodes average)")
    plt.xlabel("episodes")
    plt.ylabel("success rate (%)")
    plt.title("success rate changes")
    plt.legend()
    
    # 绘制平均步数变化
    plt.subplot(1, 3, 2)
    plt.plot(smooth_x, np.convolve(step_history, np.ones(window_size)/window_size, mode="valid"), label="average steps (100 episodes average)", color="orange")
    plt.xlabel("episodes")
    plt.ylabel("average steps")
    plt.title("average steps changes")
    plt.legend()
    
    # 绘制总奖励变化
    plt.subplot(1, 3, 3)
    plt.plot(smooth_x, np.convolve(reward_history, np.ones(window_size)/window_size, mode="valid"), label="reward (100 episodes average)", color="green")
    plt.xlabel("episodes")
    plt.ylabel("reward")
    plt.title("reward changes")
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def train():
    env = MazeGame()
    agent = QLearningAgent()
    epsilon = RLConfig.INIT_EPSILON
    
    success_history = []
    step_history = []
    reward_history = []
    
    for episode in range(RLConfig.EPISODES):
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0
        
        while not done and steps < RLConfig.MAX_STEPS:
            action = agent.choose_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.update_q(state, action, reward, next_state)
            
            state = next_state
            steps += 1
            total_reward += reward
        
        # 记录训练数据
        success_history.append(1 if reward > 0 else 0)
        step_history.append(steps)
        reward_history.append(total_reward)
        
        # 动态调整epsilon
        epsilon = max(RLConfig.MIN_EPSILON, epsilon * RLConfig.EPSILON_DECAY)
        
        # 进度显示
        if episode % 100 == 0 or episode == RLConfig.EPISODES - 1:
            avg_success = np.mean(success_history[-100:]) * 100
            print(f"Episode {episode}: 成功率={avg_success:.1f}% | 步数={steps} | 总奖励={total_reward} | ε={epsilon:.3f}")

    # 训练完成后绘制图表
    plot_training_metrics(success_history, step_history, reward_history)
    return agent

def visualize(agent):
    env = MazeGame(render=True)
    clock = pygame.time.Clock()
    
    for _ in range(5):  # 展示5次成功路径
        state = env.reset()
        done = False
        path = []
        total_reward = 0  # 用于记录每次路径的总得分
        
        while not done and len(path) < 50:
            action = np.argmax(agent.q_table[state])
            next_state, reward, done = env.step(action)
            path.append(state)
            total_reward += reward  # 累加奖励
            state = next_state
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            clock.tick(5)  # 每秒5步
        
        print(f"目标达成! 路径长度: {len(path)} | 总得分: {total_reward}")


if __name__ == "__main__":
    trained_agent = train()
    print("\n训练完成! 展示最优路径...")
    visualize(trained_agent)
