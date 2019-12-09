import numpy as np


class ReversiEnv:
    action_size = 64
    state_size = (8, 8)

    def __init__(self):
        self.state = np.zeros(self.state_size, dtype=np.float32)

    def reset(self):
        self.state = np.zeros(self.state_size, dtype=np.float32)

        return self.state

    def step(self, action, player):

        valid_actions = self.get_valid_actions(self.state, player)

        if action not in valid_actions:
            raise Exception("Invalid action", action, valid_actions)

        self.state = self.get_next_state(self.state, action, player)

        done = False
        reward = 0
        next_player = self.get_next_player(player)

        if self.is_game_over(self.state):
            done = True
            reward = self.get_winner(self.state, next_player)
        elif len(self.get_valid_actions(self.state, next_player)) == 0:
            next_player = player

        return self.state, reward, done, next_player

    @staticmethod
    def get_winner(state, player):
        unique, counts = np.unique(state, return_counts=True)
        count_dict = dict(zip(unique, counts))
        if ReversiEnv.get_next_player(player) not in count_dict:
            count_dict[ReversiEnv.get_next_player(player)] = 0
        if player not in count_dict:
            count_dict[player] = 0
        if count_dict[player] > count_dict[ReversiEnv.get_next_player(player)]:
            return 1
        elif count_dict[player] < count_dict[ReversiEnv.get_next_player(player)]:
            return -1
        else:
            return 0

    @staticmethod
    def is_game_over(state, player=1):
        first_actions = ReversiEnv.get_valid_actions(state, player)
        second_actions = ReversiEnv.get_valid_actions(state, ReversiEnv.get_next_player(player))
        return len(first_actions) == 0 and len(second_actions) == 0

    @staticmethod
    def get_next_state(state, action, player):
        action_x, action_y = action[0], action[1]
        next_state = np.copy(state)
        next_state[action_x][action_y] = player

        _, next_state = ReversiEnv.flip_direction(next_state, player, 0, action_x + 1, action_y)
        _, next_state = ReversiEnv.flip_direction(next_state, player, 1, action_x + 1, action_y + 1)
        _, next_state = ReversiEnv.flip_direction(next_state, player, 2, action_x, action_y + 1)
        _, next_state = ReversiEnv.flip_direction(next_state, player, 3, action_x - 1, action_y + 1)
        _, next_state = ReversiEnv.flip_direction(next_state, player, 4, action_x - 1, action_y)
        _, next_state = ReversiEnv.flip_direction(next_state, player, 5, action_x - 1, action_y - 1)
        _, next_state = ReversiEnv.flip_direction(next_state, player, 6, action_x, action_y - 1)
        _, next_state = ReversiEnv.flip_direction(next_state, player, 7, action_x + 1, action_y - 1)

        return next_state

    @staticmethod
    def get_next_player(player):
        return 1 if player == 2 else 2

    @staticmethod 
    def flip_direction(state, player, direction, curr_x, curr_y):
        if curr_x == 8 or curr_y == 8 or state[curr_x][curr_y] == 0:
            return False, state
        if state[curr_x][curr_y] == player:
            return True, state

        if direction == 0:
            change_color, state = ReversiEnv.flip_direction(state, player, direction, curr_x + 1, curr_y)
        elif direction == 1:
            change_color, state = ReversiEnv.flip_direction(state, player, direction, curr_x + 1, curr_y + 1) 
        elif direction == 2:
            change_color, state = ReversiEnv.flip_direction(state, player, direction, curr_x, curr_y + 1) 
        elif direction == 3:
            change_color, state = ReversiEnv.flip_direction(state, player, direction, curr_x - 1, curr_y + 1) 
        elif direction == 4:
            change_color, state = ReversiEnv.flip_direction(state, player, direction, curr_x - 1, curr_y) 
        elif direction == 5:
            change_color, state = ReversiEnv.flip_direction(state, player, direction, curr_x - 1, curr_y - 1) 
        elif direction == 6:
            change_color, state = ReversiEnv.flip_direction(state, player, direction, curr_x, curr_y - 1) 
        else:
            change_color, state = ReversiEnv.flip_direction(state, player, direction, curr_x + 1, curr_y - 1) 

        if change_color:
            state[curr_x][curr_y] = player

        return change_color, state

    @staticmethod
    def get_valid_actions(state, player):
        validMoves = []
        step = np.count_nonzero(state)

        if (step < 4):
            if (state[3][3] == 0):
                validMoves.append([3, 3])
            if (state[3][4] == 0):
                validMoves.append([3, 4])
            if (state[4][3] == 0):
                validMoves.append([4, 3])
            if (state[4][4] == 0):
                validMoves.append([4, 4])
        else:
            for i in range(8):
                for j in range(8):
                    if (state[i][j] == 0):
                        if (ReversiEnv.could_be(state, i, j, player)):
                            validMoves.append([i, j])

        return validMoves

    @staticmethod
    def could_be(state, row, col, player):
        for incx in range(-1, 2):
            for incy in range(-1, 2):
                if ((incx == 0) and (incy == 0)):
                    continue

                if (ReversiEnv.check_direction(state, row, col, incx, incy, player)):
                    return True

        return False

    @staticmethod
    def check_direction(state, row, col, incx, incy, player):
        sequence = []
        for i in range(1, 8):
            r = row + incy * i
            c = col + incx * i

            if ((r < 0) or (r > 7) or (c < 0) or (c > 7)):
                break

            sequence.append(state[r][c])

        count = 0
        for i in range(len(sequence)):
            if (player == 1):
                if (sequence[i] == 2):
                    count = count + 1
                else:
                    if ((sequence[i] == 1) and (count > 0)):
                        return True
                    break
            else:
                if (sequence[i] == 1):
                    count = count + 1
                else:
                    if ((sequence[i] == 2) and (count > 0)):
                        return True
                    break

        return False
    