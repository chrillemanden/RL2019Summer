import gym
import random as rd
import numpy as np

env = gym.make('CartPole-v0')
env.reset()

numEpochs = 2
numberRollOut = 1
learningRate = 0.8
lambda_factor = 0.5
converged = False

# Init RBF network
dimensions = [6, 6]

max_angle = 24
max_vel = 0.05
vel_centers = np.linspace(-max_vel, max_vel, dimensions[1])
angle_centers = np.linspace(-max_angle, max_angle, dimensions[0])

sigma_vel = (2*max_vel)/dimensions[1]/2
sigma_angle = (2*max_angle)/dimensions[0]/2


def initGaussianWeights():
    # list output
    output = []
    for i in range(dimensions[0]* dimensions[1]):
        output.append(rd.random())
    return output


def getOutputRBF(inputVec):
    # inputs
    vel = inputVec[0]
    angle = inputVec[1]

    # list output
    output = []

    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            output.append(np.exp(- ((vel - vel_centers[i]) ** 2) / (2 * sigma_vel ** 2) + ((angle - angle_centers[j]) ** 2) / (2 * sigma_angle ** 2)))

    return np.array(output)


def getOutput(gaussianOutputs, weights):
    output = sum(weights * gaussianOutputs)
    return output

def action_selection(inputVec, weights):
    outputRBF = getOutputRBF(inputVec)
    output = getOutput(outputRBF, weights)

    output += rd.gauss(0, 0.2)

    if (output > 0):
        action = 0
    else:
        action = 1

    return action

#def updateWeights():




weights_RBF = np.array(initGaussianWeights())

curr_iteration = 0
while not converged:
    J_delta = []
    big_Delta = [[0 for k in range(len(weights_RBF))] for k in range(numberRollOut)]
    for i in range(numberRollOut):
        roll_out = []
        state_init = env.reset()
        inputVec = np.array([state_init[3], state_init[2]])

        delta = [rd.gauss(0, 0.5) for k in range(len(weights_RBF))]

        weight_list_pm = [weights_RBF+delta , weights_RBF-delta]


        J_list = [0,0]
        for t in range(2):
            terminated = False
            state = state_init
            env.state = state_init
            weights = weight_list_pm[t]

            iteration = 0
            while not terminated:
                action = action_selection(state, weights)
                observation  = env.step(action) #terminated, reward
                state = observation[0]
                reward = observation[1]
                terminated = observation[2]
                #print(reward)
                J_list[t] += reward
                iteration += 1
            env.reset()
            print("Number of iterations: ", iteration)

        J_delta.append(J_list[0]-J_list[1])

        for j in range(len(big_Delta)):
            big_Delta[j][i] = delta[j]


    error_weights = np.array(np.dot(np.linalg.inv(np.dot(np.array(big_Delta).T, np.array(big_Delta)) + lambda_factor * np.eye(len(weights_RBF))), np.dot(np.array(big_Delta).T, np.array(J_delta))))
    M = weights_RBF.copy()
    M += 1/2 * learningRate * error_weights #np.array(np.dot(np.linalg.inv(np.dot(np.array(big_Delta).T, np.array(big_Delta)) + lambda_factor*np.eye(len(weights_RBF))),np.dot(np.array(big_Delta).T,np.array(J_delta))))
    weights_RBF = M
    curr_iteration += 1
    print("Error weights: ", error_weights)
    print("Weight: ", weights_RBF)
    if (curr_iteration > 10) :
        converged = True









        #delta =

        #for k in range()
        # do actions

#for _ in range(100):
#    env.render()
#    env.step(env.action_space.sample())

env.close()