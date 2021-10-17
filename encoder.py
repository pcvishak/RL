'''
A code to generate the mdp file given the states of 1 player and policies of the opponent in a game of Tic-Tac-Toe.
python encoder.py --policy policyfilepath --states statefilepath > mdpfile
'''


import numpy as np
import math
import argparse


def possible_actions(s):
    actions = []
    for index, i in enumerate(s):
        if i == '0':
            actions.append(index+1)
    return actions


def column_check(s):
    s_ = np.transpose(s)
    if ((list(s_[0]) == [2, 2, 2]) or (list(s_[1]) == [2, 2, 2]) or (list(s_[2]) == [2, 2, 2])):
        return 1
    if ((list(s_[0]) == [1, 1, 1]) or (list(s_[1]) == [1, 1, 1]) or (list(s_[2]) == [1, 1, 1])):
        return 2
    return 0


def row_check(s):
    if ((list(s[0]) == [2, 2, 2]) or (list(s[1]) == [2, 2, 2]) or (list(s[2]) == [2, 2, 2])):
        return 1
    if ((list(s[0]) == [1, 1, 1]) or (list(s[1]) == [1, 1, 1]) or (list(s[2]) == [1, 1, 1])):
        return 2
    return 0


def diagonal_check(s):
    if (list(np.diagonal(s)) == [2, 2, 2] or list(np.fliplr(s).diagonal()) == [2, 2, 2]):
        return 1
    if (list(np.diagonal(s)) == [1, 1, 1] or list(np.fliplr(s).diagonal()) == [1, 1, 1]):
        return 2
    return 0


def terminal(s):
    s_ = np.array([int(i) for i in s]).reshape((3, 3))
    result = column_check(s_)
    if result > 0:
        return result
    result = row_check(s_)
    if result > 0:
        return result
    result = diagonal_check(s_)
    if result > 0:
        return result
    if '0' not in s:
        return 3
    return 0


def change_state(state, index, player):
    s = list(state)
    s[index] = player
    return "".join(s)


def fn(filename_states, filename_policies):
    #filename_states = 'pa2_base/data/attt/states/states_file_p2.txt'
    with open(filename_states) as f:
        lines_states = f.readlines()

    #filename_policies = 'pa2_base/data/attt/policies/p1_policy1.txt'
    with open(filename_policies) as f:
        lines_policies = f.readlines()

    states = []
    for line in lines_states:
        states.append(line.split('\n')[0])

    index_states = {}
    for index, s in enumerate(states):
        index_states[s] = index

    opponent_states = []
    for index, line in enumerate(lines_policies):
        if (index != 0):
            opponent_states.append(line.split(' ')[0])

    index_opponent_states = {}
    for index, s in enumerate(opponent_states):
        index_opponent_states[s] = index

    pis = []
    for index, line in enumerate(lines_policies):
        l = []
        if (index != 0):
            line = line.split('\n')[0]
            l.append(line.split(' ')[1:])
            pis.append(l)
        else:
            opponent = line.split('\n')[0]

    if opponent == '1':
        player = '2'
    else:
        player = '1'

    num_states = len(states) + 2
    num_actions = 9
    end = -1

    T = np.zeros((num_states, num_actions, num_states))
    R = np.zeros((num_states, num_actions, num_states))

    for state in states:
        actions = possible_actions(state)

        for action in actions:

            # Agent playing with action
            new_s = state
            new_s = change_state(new_s, action - 1, player)

            is_terminal = terminal(new_s)

            if (is_terminal > 0):
                s = index_states[state]
                #s_ = num_states-1
                #T[s][action-1][s_] = 1.0
                if (is_terminal == int(player)):

                    s_ = num_states-2
                    T[s][action-1][s_] = 1.0
                    R[s][action-1][s_] = 1
                else:

                    s_ = num_states-1
                    T[s][action-1][s_] = 1.0
                    R[s][action-1][s_] = 0

            else:
                # Opponent playing
                opponent_actions = possible_actions(new_s)
                for oa in opponent_actions:
                    temp_state = new_s
                    temp_state = change_state(temp_state, oa - 1, opponent)

                    is_t = terminal(temp_state)
                    pi = pis[index_opponent_states[new_s]]
                    if (is_t > 0):
                        s = index_states[state]
                        #s_ = num_states-1
                        #T[s][action-1][s_] += float(pi[0][oa - 1])
                        if (is_t == int(player)):

                            s_ = num_states-2
                            T[s][action-1][s_] += float(pi[0][oa - 1])
                            R[s][action-1][s_] = 1
                        else:

                            s_ = num_states-1
                            T[s][action-1][s_] += float(pi[0][oa - 1])
                            R[s][action-1][s_] = 0

                    else:

                        s = index_states[state]
                        s_ = index_states[temp_state]

                        T[s][action-1][s_] += float(pi[0][oa - 1])
                        R[s][action-1][s_] = 0

    '''
    with open('mdpfile.txt', 'a') as the_file:
        the_file.write("numStates" + " " + str(num_states) + "\n")
        the_file.write("numActions" + " " + str(num_actions) + "\n")
        the_file.write("end" +" " +str(len(states)) +" " +str(len(states)) +"\n")
        for i in range(num_states):
            for j in range(num_actions):
                for k in range(num_states):
                    if (T[i][j][k] != 0.0): 
                        the_file.write("transition" + " " + str(i) + " " + str(j) + " " + str(k) + " " + str(R[i][j][k]) + " " + str(T[i][j][k]) + "\n")
        the_file.write("mdptype episodic\n")
        the_file.write("discount 1.0\n")
    '''
    '''
    print("numStates" + " " + str(num_states) + "\n")
    print("numActions" + " " + str(num_actions) + "\n")
    print("end" +" " +str(len(states)) +" " +str(len(states)) +"\n")
    for i in range(num_states):
        for j in range(num_actions):
            for k in range(num_states):
                if (T[i][j][k] != 0.0): 
                    print("transition" + " " + str(i) + " " + str(j) + " " + str(k) + " " + str(R[i][j][k]) + " " + str(T[i][j][k]) + "\n")
    print("mdptype episodic\n")
    print("discount 1.0\n")
    '''
    print("numStates" + " " + str(num_states))
    print("numActions" + " " + str(num_actions))
    print("end" + " " + str(len(states)-1) + " " + str(len(states)))
    for i in range(num_states):
        for j in range(num_actions):
            for k in range(num_states):
                if (T[i][j][k] != 0.0):
                    print("transition" + " " + str(i) + " " + str(j) + " " +
                          str(k) + " " + str(R[i][j][k]) + " " + str(T[i][j][k]))
    print("mdptype episodic")
    print("discount 1.0")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy')
    parser.add_argument('--states')
    args = parser.parse_args()
    fn(args.states, args.policy)


if __name__ == "__main__":
    main()
