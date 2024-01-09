# Taskmaster

With this project I wanted to try in how AlphaZero type algorithms may be augmented to create systems that solve combinatorial puzzles. So far, I have focused on 2x2x2 and 3x3x3 Rubik's cubes.

## Differences to AlphaZero

There are some differences in my implementation to the original AlphaZero.

    * Rather than estimating the policy after the MTCS, I estimate the expected reward of each action. The MTCS refines this estimate through rollouts and we take the action with the highest reward according to this refined estimate.
    * In each rollout, we find a leaf of the tree from the current state by choosing at each node an action a so that 
    estimated_reward(a)+standard_error_of_estimate(a) is maximized. Both estimated reward and standard error are estimated through a combination of a confidence estimate of the reward estimate network and the result of previous rollouts.
    * In order to be able to batch the neural network calls, I first find a leaf in the tree of each problem in the batch and then let
    keras compute the values, expected rewards and confidences for all of these leaves simultaneously.

## DeepCube and reversible steps

There is a [paper](https://arxiv.org/pdf/1805.07470.pdf) by McAleer et al. where they use AlphaZero style NN-guided MCTS to solve Rubik's cubes. An important difference however is that they train the NN independently before they apply the MCTS. As far as I can tell, their way of training the NN relies crucially on them already knowing at least one solution path for any cube that they have scrambled, namely reversing the scrambling steps.

I found a [repository](https://github.com/kongaskristjan/rubik) where this idea is pushed even further: they don't use any MCTS and just teach the NN to reverse scrambling steps, and apparently that is enough to teach it to solve the cube.

## The teacher

One important difference between the tasks for which AlphaZero was developed (e.g. chess, go, etc.) and the combinatorial puzzles that Taskmaster is concerned with is the extreme sparsity of rewards in the latter. For example, if we try to let the system learn from completely scrambled Rubik's cubes, that will never get anywhere. One solution is to start by manually feeding the system easy cubes at first and increasing the difficulty over time, which is what I have done so far. In the future, I would like create a second, semi-adiversarial "teacher" system, that constructs the problems of a given puzzle type in a way that maximizes the uncertainty of whether the student can solve the problem.

## TODO

    * Try out other NN architectures with fewer parameters
    * Reproduce DeepCube or the scrambling reversal and see which NN architectures work best for them
    * Adapt training of NN so that loss is ignored for reward estimates of actions that have not been explored (sufficiently)
    * Implement other puzzles
    * Build capability for two player games to see whether my changes to vanilla AlphaZero are valid
    * Implement the teacher