# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        # initialize values to 0
        # if we haven't seen them before
        self.qValues = util.Counter() # a Counter is a dict with default 0

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        
        """
        Make sure that in your computeValueFromQValues and computeActionFromQValues
        functions, you only access Q values by calling getQValue . This abstraction
        will be useful for question 10 when you override getQValue to use features
        of state-action pairs rather than state-action pairs directly.
        """
        # self.qValues are initialized to util.Counter
        # where values are 0 as default
        # so 0.0 is returned if we have never seen a state
        # or the Q node value otherwise
        return self.qValues[(state, action)]

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        
        legalActions = self.getLegalActions(state)
        # return a value of 0.0 if you are in the terminal state
        # there are no legal actions in the terminal state
        if not legalActions:
            return 0.0
        """
        Make sure that in your computeValueFromQValues and computeActionFromQValues
        functions, you only access Q values by calling getQValue . This abstraction
        will be useful for question 10 when you override getQValue to use features
        of state-action pairs rather than state-action pairs directly.
        Note: For computeActionFromQValues, you should break ties randomly for
        better behavior. The random.choice() function will help. In a particular
        state, actions that your agent hasn't seen before still have a Q-value,
        specifically a Q-value of zero, and if all of the actions that your agent
        has seen before have a negative Q-value, an unseen action may be optimal.
        """
        # create list of qValus for each legal action
        qValues = [self.getQValue(state, action) for action in legalActions]
        """
        got this idea for finding all positions of max value in a list
        from this Stack Overflow post:
        
        https://stackoverflow.com/questions/3989016/how-to-find-all-positions-of-the-maximum-value-in-a-list
        """
        # break ties randomly for better behavior
        # the ranom.choice() function will help
        maxqValue = max(qValues)
        # find all indices where there is a max value
        maxIndices = [i for i, j in enumerate(qValues) if j == maxqValue]
        # use the random.choice function as recommended
        randomIndex = random.choice(maxIndices)
        # return the max qValue at that random index
        return qValues[randomIndex]
        
        

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        # if there are no legal actions
        # which is the case at the terminal state,
        # you should return None
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        """
         Make sure that in your computeValueFromQValues and
         computeActionFromQValues functions, you only access Q values by calling
         getQValue . This abstraction will be useful for question 10 when you
         override getQValue to use features of state-action pairs rather than
         state-action pairs directly.
        """
        # util.Counter() is dictionary with default value 0
        qActions = util.Counter()
        # for each possible legal action,
        # have dictionary where an action is the key
        # the qValues are the dictionary values
        # fill dictionary with qValue associated with that state and action
        for action in legalActions:
            qActions[action] = self.getQValue(state, action)
        # util.argMax() returns the key with the highest value.
        # in this case the action with highest Q Value
        return qActions.argMax()

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)
        action = None
        "*** YOUR CODE HERE ***"
        # action is initialized to None
        # if there are no legal actions,
        # which is the case at the terminal state,
        # return None as the action
        """
        Complete your Q-learning agent by implementing epsilon-greedy action
        selection in getAction, meaning it chooses random actions an epsilon
        fraction of the time, and follows its current best Q-values otherwise. Note
        that choosing a random action may result in choosing the best action - that
        is, you should not choose a random sub-optimal action, but rather any
        random legal action.
        You can choose an element from a list uniformly at random by calling the
        random.choice function. You can simulate a binary variable with probability
        p of success by using util.flipCoin(p), which returns True with probability
        p and False with probability 1-p.
        """
        if legalActions:
            # with probability self.epsilon,
            # take a random action and take the best policy action otherwise
            if util.flipCoin(self.epsilon):
                # to pick randomly from a list, use random.choice(list)
                action = random.choice(legalActions)
            # othewise ake the best policy action
            else:
                action = self.computeActionFromQValues(state)
            

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        
        # alpha is the learning rate
        alpha = self.alpha
        # Q(state,action) stored in self.qValues
        oldQValue = self.getQValue(state, action)
        
        gamma = self.discount
        
        # max_action Q(nextState,action) where the max is over legal actions
        successorQValue = self.computeValueFromQValues(nextState)
        
        sample = reward + (gamma * successorQValue)
        self.qValues[(state, action)] = (1 - alpha) * oldQValue + alpha * sample 
        

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        
        """
        Note: Approximate Q-learning assumes the existence of a feature function
        f(s,a) over state and action pairs, which yields a vector f1(s,a) ..
        fi(s,a) .. fn(s,a) of feature values. We provide feature functions for you
        in featureExtractors.py. Feature vectors are util.Counter (like a
        dictionary) objects containing the non-zero pairs of features and values;
        all omitted features have value zero.
        By default, ApproximateQAgent uses the IdentityExtractor, which assigns a
        single feature to every (state,action) pair.
        """
        """
        Multiplying two counters gives the dot product of their vectors where
        each unique label is a vector element.
        Get Q(s,a) by multiplying the weight and feature counters together.
        """
        return self.weights * self.featExtractor.getFeatures(state, action)

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        
        
        """
        Note that the difference term
        is the same as in normal Q-learning,
        and r is the experienced reward.
        """
        """
         In your code, you should implement the weight vector as a dictionary
         mapping features (which the feature extractors will return) to weight
         values. You will update your weight vectors similarly to how you updated
         Q-values:
        """
        gamma = self.discount
        successorQValue = self.computeValueFromQValues(nextState)
        sample = reward + (gamma * successorQValue)
        
        qValue = self.getQValue(state, action)
        difference = sample - qValue
        # retrieve vector of feature values
        features = self.featExtractor.getFeatures(state, action)
        
        
        # alpha is the learning rate
        alpha = self.alpha
        # the weight vector is a dictionary mapping features to weight values
        for feature in features:
            """
            To update weights, add to the current weight_i,
            alpha * difference * f_i(s,a)
            """
            self.weights[feature] += alpha * difference * features[feature]
        

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "*** YOUR CODE HERE ***"
            #print("Weights")
            print(f"Weights: {self.weights}")
            
