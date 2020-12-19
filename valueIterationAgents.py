# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        
        """
        create Counter for values
        A Counter is a dict with default 0.
        This initializes all values to zero to start.
        My first thought was to update values on self.values directly,
        but it is important to have all values initialized
        to the same value to start.
        """
        values = util.Counter()
        
        
        # referenced Value Iteration pseudocode from these slides:
        # https://www.cs.ubc.ca/~kevinlb/teaching/cs322%20-%202007-8/Lectures/DT4.pdf
        # run this for number of iterations passed into class
        for k in range(self.iterations):
            # get states for the mdp
            states = self.mdp.getStates()
            # for each state in s
            for state in states:
                # get list of possible actions from that state 
                actions = self.mdp.getPossibleActions(state)
                """
                make sure to handle case where state has no available actions in MDP
                if there aren't any available actions from a state, the value won't be updated
                continue statement returns control to the top of the loop to run another iteration
                referenced this source on loop conrol:
                https://www.tutorialspoint.com/python/python_loop_control.htm
                """
                if not actions:
                    continue
                # for each action in possible actions, compute the Q value for
                # that state and action
                # V*(s) = max action Q*(s, a)
                # return the max Q-value
                maxValue = max([self.computeQValueFromValues(state, action) for action in actions])
                
                # set value for state key to the max Q-value
                values[state] = maxValue
                
            # Use the values dictionary to update self.values
            # Approach to use update came from this Stack Overflow post:
            # https://stackoverflow.com/questions/21719842/copying-a-key-value-from-one-dictionary-into-another
            self.values.update(values)
                
    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        
        # transitions is a list of (nextState, prob) pairs
        transitions = self.mdp.getTransitionStatesAndProbs(state, action)
        
        # initialize a qvalue accumulator
        qvalue = 0
        # assign each part of the tuple in transitions to its own variable
        # for each entry in the transitions list
        for nextState, probability in transitions:
            # mdp.getReward(state, action, nextState)
            reward = self.mdp.getReward(state, action, nextState)
            gamma = self.discount
            # find the value from the successor state
            successorValue = self.values[nextState]
            # use Bellman update equation
            # this will return the sum for all successor states
            # so add to the accumulator
            qvalue += probability * (reward + gamma * successorValue)
        # return the sum of values for all successor states
        return qvalue
        
    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # if there are no legal actions, which is the case
        # at the terminal state, return None
        if self.mdp.isTerminal(state):
            return None
        
        # returns list of possible legal actions in a state
        actions = self.mdp.getPossibleActions(state)
        
        """
        The best action is going to be the action associated with the highest Q-value.
        I viewed this Quora answer for," How do I find the largest and smallest number
        in an unsorted integer array? to inform my strategy for this.
        https://www.quora.com/How-do-I-find-the-largest-and-smallest-number-in-an-unsorted-integer-array
        Since a Q-Value is for a state, action pair,
        keeping track of the maxValue in this manner, makes it easy to also track the best action at the same time.
        """
        # initialize maxValue to smallest possible value
        maxValue = float("-inf")
        # intialize best action to None
        bestAction = None
        # look at every possible QValue and compare it to the max value
        # loop over every possible legal action
        for action in actions:
            # compute the QValue for that state, action pair
            qValue = self.computeQValueFromValues(state, action)
            # if the current maxValue is less than this qValue
            # save the new largest value
            # use a strictly less than so that ties will be broken
            # by returning the first occurrence with that value in the event of a tie
            if maxValue < qValue:
                maxValue = qValue
                # if the maxValue is updated, it means that the best action also needs to be updated
                bestAction = action
        
        return bestAction
            

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        # the code for this function is similar to runValueIteration
        # in the ValueIterationAgent class
        # this time, update only one state in each iteration, as opposed to doing a batch-style update
        
        # run this for number of iterations passed into class
        for k in range(self.iterations):
            """
            each iteration updates the value of only one state
            which cycles through the states list
            Keep going until you have updated the value of each state once,
            then start back at the first state for the subsequent iteration
            """
        
            # returns a list of all states in the mdp
            states = self.mdp.getStates()
            # use modulo to select a state
            # and then start back at the first state
            # when the value of each state has been updated once
            state = states[k % len(states)]
            
            # if the chosen state is terminal,
            # nothing happens in that iteration
            if self.mdp.isTerminal(state):
                continue
           
                    
            actions = self.mdp.getPossibleActions(state)
            # for each action in possible actions, compute the Q value for
            # that state and action pair
            # return the max Q-value
            maxValue = max([self.computeQValueFromValues(state, action) for action in actions])
                
            # update self.values dict with maxValue
            self.values[state] = maxValue
                
            

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        
        states = self.mdp.getStates()
        """
        First, we define the predecessors of a state s as all states that have a nonzero probability of reaching s by taking some action a.
        When you compute predecessors of a state, make sure to store them in a set, not a list, to avoid duplicates.
        
        *compute predecessors of all states
        *store predecessors in a dictionary
        where keys are states
        values are sets of the predecesors of a state
        """
        # initialize an empty predecessor dictionary
        predecessors = {}
        for state in states:
            # terminal states don't have legal actions that can be taken from them
            # so this needs to be limited to non-terminal states
            if self.mdp.isTerminal(state):
                continue
            
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                # get transitions
                # Returns list of (nextState, prob) pairs
                # the transitions help connect states and successor states
                transitions = self.mdp.getTransitionStatesAndProbs(state, action)
                # unpack the tuple in transitions list into its own variables
                for nextState, probability in transitions:
                    """
                    There are two cases. Case 1: a next state isn't in the predecessors
                    dictionary yet. In that case, we need to add a key for next state
                    to the predecessors dictionary with a set that includes state.
                    Case 2: a next state is already in the predecessors dictionary.
                    Add state to the set of predecessors for next state.
                    We are using a set instead of a list because a set will avoid duplicates.
                        
                    """
                    if nextState not in predecessors:
                        # add nextState as a key with
                        # value that is a set with state in it 
                        # source for how to initialize/update set:
                        # https://www.programiz.com/python-programming/set
                        predecessors[nextState] = {state}
                    else:
                        predecessors[nextState].add(state)
                    
        """
        A note in the project directions:
        Please use util.PriorityQueue in your implementation.
        The update method in this class will likely be useful; look at its documentation.
        """
        # initialize an empty priority queue
        pq = util.PriorityQueue()
        """
        note: to make the autograder work for this question, you must iterate over states in the order returned by self.mdp.getStates())
        """
        # for each nonterminal state s
        for state in states:
            # if the chosen state is terminal,
            # nothing happens in that iteration
            if self.mdp.isTerminal(state):
                continue
            """
            Find the absolute value of the difference between the current value of
            s in self.values and the highest Q-value across all possible actions
            from s (this represents what the value should be); call this number
            diff. Do NOT update self.values[s] in this step.
            """
            # the current value of s in self.values
            currentValue = self.values[state]
            # get all possible actions from s
            actions = self.mdp.getPossibleActions(state)
            # get the highest Q-value across all possible actions from s
            # this represents what the value should be
            # for each action in possible actions, compute the Q value for
            # that state and action
            # return the max Q-value
            maxValue = max([self.computeQValueFromValues(state, action) for action in actions])
            # find the absolue value of the difference between the current value of s
            # in self.values and the highest Q-Value across all possible actions
            # from s. This represents what the value should be.
            # call this number diff
            diff = abs(currentValue - maxValue)
            """
            Push s into the priority queue with priority -diff (note that this is
            negative). We use a negative because the priority queue is a min heap,
            but we want to prioritize updating states that have a higher error.
            """
            pq.push(state, -diff)
            
        # For iteration in 0, 1, 2, ..., self.iterations - 1, do:
        # run this for number of iterations passed into class
        for k in range(self.iterations):
            # If the priority queue is empty, then terminate.
            if pq.isEmpty():
                break
            
            # Pop a state s off the priority queue.
            poppedState = pq.pop()
            # Update s's value (if it is not a terminal state) in self.values.
            if not self.mdp.isTerminal(poppedState):
                # get all possible actions from s
                actions = self.mdp.getPossibleActions(poppedState)
                # get the highest Q-value across all possible actions from s
                # this represents what the value should be
                # for each action in possible actions, compute the Q value for
                # that state and action
                # return the max Q-value
                maxValue = max([self.computeQValueFromValues(poppedState, action) for action in actions])
                self.values[poppedState] = maxValue
            # return the set of predecessors for s
            sPredecessors = predecessors[poppedState]
                
            # For each predecessor p of s, do
            for predecessor in sPredecessors:
                """
                Find the absolute value of the difference between the current
                value of p in self.values and the highest Q-value across all
                possible actions from p (this represents what the value should
                be); call this number diff. Do NOT update self.values[p] in
                this step.
                    
                """
                currentValue = self.values[predecessor]
                actions = self.mdp.getPossibleActions(predecessor)
                      
                maxValue = max([self.computeQValueFromValues(predecessor, action) for action in actions])
                diff = abs(maxValue - currentValue)
                """
                If diff > theta, push p into the priority queue with
                priority -diff (note that this is negative), as long as it
                does not already exist in the priority queue with equal or
                lower priority. As before, we use a negative because the
                priority queue is a min heap, but we want to prioritize
                updating states that have a higher error.
                """
                if diff > self.theta:
                    pq.update(predecessor, -diff)
            
                
                
            
