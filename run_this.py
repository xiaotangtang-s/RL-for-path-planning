from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    for episode in range(5000):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 10000) and (step % 5 == 0):    # 记忆量设置为2000
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.1,
                      reward_decay=0.9,
                      e_greedy=0.8,
                      replace_target_iter=2000,
                      memory_size=50000,
                      output_graph=True
                      )
    env.after(100, run_maze)   # after()实现简单的定时器功能，after(100,run_maze)表示每隔0.1S执行一次run_maze函数
    env.mainloop()
    RL.plot_cost()