'''
Code for finding the Value of the states and the optimal actions,
Methods used include Value Iteration, Howards Policy iteration and Linear Programming.
python planner.py --mdp continuing-mdp-50-20.txt --algorithm vi
python planner.py --mdp episodic-mdp-50-20.txt --algorithm hpi
python planner.py --mdp episodic-mdp-50-20.txt --algorithm lp
'''

import numpy as np
import math
import argparse
from pulp import *


def value_iteration(filename):

    #filename = 'mdpfile.txt'
    with open(filename) as f:
        lines = f.readlines()

    num_states = [int(s) for s in lines[0].split() if s.isdigit()][0]
    num_actions = [int(s) for s in lines[1].split() if s.isdigit()][0]

    R = np.zeros((num_states, num_actions, num_states))
    T = np.zeros((num_states, num_actions, num_states))

    for line in lines:
        t = line.split(' ')
        if (t[0] == 'transition'):
            T[int(t[1])][int(t[2])][int(t[3])] = float(t[-1])
            R[int(t[1])][int(t[2])][int(t[3])] = float(t[-2])
        if (t[0] == 'mdptype'):
            mdptype = t[-1].split('\n')[0]
        if (t[0] == 'discount'):
            discount = float(t[-1])
        if (t[0] == 'end'):
            terminal_states = [int(s) for s in line.split() if s.isdigit()]

    V = np.array([0.0] * num_states)
    Prev = np.array([-10.0] * num_states)
    pi = np.array([0] * num_states)
    t = 0

    while not(all(math.isclose(V[s], Prev[s], abs_tol=np.finfo(float).eps) for s in range(num_states))):

        np.copyto(Prev, V)
        temp = np.ones((num_states, num_actions, num_states))
        V_ = temp * V
        V_ = (discount * V_)
        result = T * (R + V_)
        V = np.max(np.sum(result, 2), 1)
        pi = np.argmax(np.sum(result, 2), 1)

    for s in range(num_states):
        print(format(V[s], '.6f') + "\t" + str(pi[s]) + "\n")


def howards_policy_iteration(filename):

    #filename = 'mdpfile.txt'
    with open(filename) as f:
        lines = f.readlines()

    num_states = [int(s) for s in lines[0].split() if s.isdigit()][0]
    num_actions = [int(s) for s in lines[1].split() if s.isdigit()][0]

    R = np.zeros((num_states, num_actions, num_states))
    T = np.zeros((num_states, num_actions, num_states))

    for line in lines:
        t = line.split(' ')
        if (t[0] == 'transition'):
            T[int(t[1])][int(t[2])][int(t[3])] = float(t[-1])
            R[int(t[1])][int(t[2])][int(t[3])] = float(t[-2])
        if (t[0] == 'mdptype'):
            mdptype = t[-1].split('\n')[0]
        if (t[0] == 'discount'):
            discount = float(t[-1])
        if (t[0] == 'end'):
            terminal_states = [int(s) for s in line.split() if s.isdigit()]

    V = [0] * num_states
    Q = np.zeros((num_states, num_actions))
    pi = [0] * num_states
    count = 0

    while (1):
        # print(count)
        count = count + 1
        IA = []
        for i in range(num_states):
            IA.append(set())
        IS = set()

        for s in range(num_states):

            if (s in terminal_states):
                continue
            for a in range(num_actions):
                sum_ = 0
                for s_ in range(num_states):
                    sum_ += T[s][a][s_]*(R[s][a][s_] + discount*V[s_])
                Q[s][a] = sum_
            for a in range(num_actions):
                if (Q[s][a] > V[s]):
                    IA[s].add(a)
                    IS.add(s)

        if (len(IS) == 0):
            break

        for s in IS:
            V[s] = Q[s][min(IA[s])]
            pi[s] = min(IA[s])

    for s in range(num_states):
        print(str(format(V[s], '.6f')) + "\t" + str(pi[s]) + "\n")


def LP(filename):

    #filename = 'mdpfile.txt'
    with open(filename) as f:
        lines = f.readlines()

    num_states = [int(s) for s in lines[0].split() if s.isdigit()][0]
    num_actions = [int(s) for s in lines[1].split() if s.isdigit()][0]

    R = np.zeros((num_states, num_actions, num_states))
    T = np.zeros((num_states, num_actions, num_states))

    for line in lines:
        t = line.split(' ')
        if (t[0] == 'transition'):
            T[int(t[1])][int(t[2])][int(t[3])] = float(t[-1])
            R[int(t[1])][int(t[2])][int(t[3])] = float(t[-2])
        if (t[0] == 'mdptype'):
            mdptype = t[-1].split('\n')[0]
        if (t[0] == 'discount'):
            discount = float(t[-1])
        if (t[0] == 'end'):
            terminal_states = [int(s) for s in line.split() if s.isdigit()]

    # Create the 'prob' variable to contain the problem data
    prob = LpProblem("The MDP", LpMaximize)
    V = []
    for s in range(num_states):
        x = LpVariable("value of state" + str(s), 0)
        V.append(x)
    prob += -sum(V), "Sum of state Values"
    for s in range(num_states):
        for a in range(num_actions):
            sum_ = 0
            for s_ in range(num_states):
                sum_ += T[s][a][s_]*(R[s][a][s_] + discount*V[s_])
            prob += V[s] >= sum_, "Constraint " + str(s) + " " + str(a)
    prob.writeLP("Mdp_model.lp")
    # prob.solve(GLPK_CMD(msg=False))
    prob.solve(PULP_CBC_CMD(msg=False))
    #print("Status:", LpStatus[prob.status])

    pi = [0] * num_states
    for s in range(num_states):
        max_value = -1000000
        for a in range(num_actions):
            sum_ = 0
            for s_ in range(num_states):
                sum_ += T[s][a][s_]*(R[s][a][s_] + discount*V[s_].varValue)
            if (sum_ > max_value):
                max_value = sum_
                max_action = a
        pi[s] = max_action

    for s in range(num_states):
        print(format(V[s].varValue, '.6f') + "\t" + str(pi[s]) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp')
    parser.add_argument('--algorithm', default='vi')
    args = parser.parse_args()

    if (args.algorithm == 'vi'):
        value_iteration(args.mdp)
    elif (args.algorithm == 'hpi'):
        howards_policy_iteration(args.mdp)
    elif (args.algorithm == 'lp'):
        LP(args.mdp)


if __name__ == "__main__":
    main()
