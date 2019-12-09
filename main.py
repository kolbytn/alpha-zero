import numpy as np

from ReversiEnv import ReversiEnv
from AlphaAgent import AlphaAgent


if __name__ == '__main__':
    iterations = 1000
    games_per_iter = 100
    games_per_compare = 40
    update_tresh = .6
    num_sims = 20
    cpuct = 1
    model_path = 'weights_/'

    env = ReversiEnv()
    agent = AlphaAgent(num_sims, cpuct, model_path=model_path)
    new_agent = AlphaAgent(num_sims, cpuct)

    for i in range(iterations):
        memory = []
        for j in range(games_per_iter):

            state = env.reset()
            player = j % 2 + 1
            done = False
            rollout = []

            while not done:
                
                action, pi = agent.get_action(state, player)

                for r in range(1, 5):
                    rollout.append((np.rot90(state, r).copy(), player, np.rot90(np.reshape(pi, state.shape), r).flatten()))
                    rollout.append((np.fliplr(np.rot90(state, r)).copy(), player, np.fliplr(np.rot90(np.reshape(pi, state.shape), r)).flatten()))

                state, reward, done, player = env.step(action, player)

            rollout = [(x[0], x[1], x[2], np.array([reward], dtype=np.float32) if x[1] != player else np.array([-reward], dtype=np.float32)) for x in rollout]
            memory.extend(rollout)

            print('iter {} game {} reward {}'.format(i, j, reward if player == 1 else -reward))


        new_agent.learn(memory)

        new_agent_wins = 0
        agent_wins = 0
        for j in range(games_per_compare):

            state = env.reset()
            player = j % 2 + 1
            done = False

            while not done:

                if player == 1:
                    action, pi = new_agent.get_action(state, player)
                else:
                    action, pi = agent.get_action(state, player)

                state, reward, done, player = env.step(action, player)

            if reward == 1 and player == 1 or reward == -1 and player == 2:
                new_agent_wins += 1
            elif reward == 1 and player == 2 or reward == -1 and player == 1:
                agent_wins += 1

        if new_agent_wins + agent_wins > 0 and new_agent_wins / (new_agent_wins + agent_wins) > update_tresh:
            print('UPDATE AGENT: {} wins'.format(new_agent_wins / (new_agent_wins + agent_wins)))
            agent.net.load_state_dict(new_agent.net.state_dict())
            agent.save_model(i)
        else:
            print('KEEP AGENT: {} wins'.format(new_agent_wins / (new_agent_wins + agent_wins) if new_agent_wins + agent_wins > 0 else -1))
            new_agent.net.load_state_dict(agent.net.state_dict())
