# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
       

        # turn boolean grid of food positions into a list
        # this will be a list of only the positions
        # where there is food
        foodPositions = newFood.asList()
        """
        calculate the Manhattan Distance
        from Pacman's position to every element in the list of food positions
        I looked at this source to figure out how to do this with list comprehension
    https://www.kite.com/python/answers/how-to-apply-a-function-to-a-list-in-python
        """
        foodDistance = [util.manhattanDistance(element, newPos) for element in foodPositions]
        # if there is still food left, the closest food pellet
        # is the one with the min distance
        if foodDistance:
            closestFood = min(foodDistance)
        # if all food is gone, set closest food to 0
        # so it doesn't have an impact
        else:
            closestFood = 1
        
        # number of food pellets left
        numFood = successorGameState.getNumFood()
        
        # if Pacman eats a capsule, the ghost gets scared
        # if Pacman eats a scared ghost, he gets 200 points.
        # calculae Manhattan Distance from Pacman to nearest capsule.
        capsulePositions = successorGameState.getCapsules()
        capsuleDistance = [util.manhattanDistance(element, newPos) for element in capsulePositions]
        if capsuleDistance:
            closestCapsule = min(capsuleDistance)
        else:
            closestCapsule = 0.01
        
        # if Pacman is close the capsule
        # the capsule will impact his action's
        # otherwise, his focus will be on getting the food pellets
        if closestCapsule < 2:
            capsuleScore = 1
        else:
            capsuleScore = 0
        
        # give Pacman incentive to actually eat the capsule
        # rather than get stuck thrashing close to the capsule
        numCapsules = len(capsulePositions)
        if numCapsules == 0:
            numCapsulesScore = 50
        else:
            numCapsulesScore = 0
        
        # calculae Manhattan Distance from Pacman to ghost
        ghostPositions = successorGameState.getGhostPositions()
        ghostDistance = [util.manhattanDistance(element, newPos) for element in ghostPositions]
        # focus on the closest ghost
        if ghostDistance:
            closestGhost = min(ghostDistance)
        else:
             closestGhost = 1
        
        # figure out whether the ghost is scared
        # or not using the scared timer
        if max(newScaredTimes) > 0:
            scaredGhost = True
            activeGhost = False
        else:
            scaredGhost = False
            activeGhost = True
        
        # if the ghost is scared,
        # Pacman needs o chase the ghost
        if scaredGhost:
            scaredGhostScore = 1/closestGhost
        else:
            scaredGhostScore = 0
    
        # otherwise Pacman needs to avoid the ghost
        if activeGhost and closestGhost <= 1:
            activeGhostScore = -10
        else: activeGhostScore = 0
       
        # linear value function using a feature representation
        score = successorGameState.getScore() + (1 * (1/closestFood)) + (10 * capsuleScore) + (10 * numCapsulesScore) + scaredGhostScore + activeGhostScore
        
        return score
        
        

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        # agentIndex 0 is pacman
        # pacman is the root with depth 0
        agentIndex = 0
        currentDepth = 0
        # call maxValue for the Pacman root
        utility = self.maxValue(gameState, agentIndex, currentDepth)
        # utility is a tuple with action and max value
        # return the action part of the tuple
        return utility[1]
    
    def maxValue(self, gameState, agentIndex, currentDepth):
        # if the state is a terminal state:
        # return the state's utility
        if self.depth==currentDepth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), Directions.STOP)
    
        # we find what legal actions Pacman can take
        legalActions = gameState.getLegalActions(agentIndex)
        # initialize v to negative infinity
        v = (float("-inf"), Directions.STOP)
    
        # loop over each successor of state
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # maxValue and minValue recursively call each other
            # call minValue with incremented agentIndex to get v'
            vPrime = self.minValue(successor, agentIndex+1, currentDepth)
       
            # set v to the max(v, v')
            # sets the action part of the tuple as well
            if vPrime[0] > v[0]:
                v = (vPrime[0], action)
        
        # return v, tuple with value and action
        return v
    
    def minValue(self, gameState, agentIndex, currentDepth):
        # if the state is a terminal state:
        # return the state's utility
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), Directions.STOP)
    
    
        # retrieve the number of agents,
        # this includes all ghosts and Pacman
    
        numAgents = gameState.getNumAgents()
        # the next agent after every ghost except for the
        # last ghost is a MIN agent
        # for a ghost that isn't the one with the highest index
        if (agentIndex < numAgents-1):
            # initialize float to negative infinity
            v = (float("inf"), Directions.STOP)
        
            # get the legal actions the ghost can take
            legalActions = gameState.getLegalActions(agentIndex)
        
            # for each successor of state
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # call minValue with incremented index to get v'
                vPrime = self.minValue(successor, agentIndex+1, currentDepth)
            
                # v = min(v, v')
                if vPrime[0] < v[0]:
                    v = (vPrime[0], action)
            
            # return value, action tuple
            return v
        
        # the ghost with the highest index has Pacman
        # as the next agent
        elif (agentIndex == numAgents-1):
            # initialize v to infinity
            v = (float("inf"), Directions.STOP)
        
            # get the legal actions the ghost can take
            legalActions = gameState.getLegalActions(agentIndex)
        
            # for each successor of state
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # call maxValue function with Pacman's agentIndex 0
                # a single depth ply has Pacman and each ghost's actions
                # this means that this is where we increment the depth
                vPrime = self.maxValue(successor, 0, currentDepth+1)
            
                # v = min(v, v')
                if vPrime[0] < v[0]:
                    v = (vPrime[0], action)
            
            # return value, action tuple
            return v
        
        
                    
        
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # agentIndex 0 is pacman
        # pacman is the root with depth 0
        agentIndex = 0
        currentDepth = 0
        # alpha is MAX's best option on path to root
        # initialize alpha to negative infinity
        alpha = (float("-inf"), Directions.STOP)
        # beta is MIN's best option on path to root
        # initialize beta to positive infinity
        beta = (float("inf"), Directions.STOP)
        # call maxValue for the Pacman root
        utility = self.maxValue(gameState, agentIndex, currentDepth, alpha, beta)
        # utility is a tuple with action and max value
        # return the action part of the tuple
        return utility[1]
        
    
    def maxValue(self, gameState, agentIndex, currentDepth, alpha, beta):
        # if the state is a terminal state:
        # return the state's utility
        if self.depth==currentDepth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), Directions.STOP)
        
        # we find what legal actions Pacman can take
        legalActions = gameState.getLegalActions(agentIndex)
        # initialize v to negative infinity
        v = (float("-inf"), Directions.STOP)
        
        # loop over each successor of state
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # maxValue and minValue recursively call each other
            # call minValue with incremented agentIndex to get v'
            vPrime = self.minValue(successor, agentIndex+1, currentDepth, alpha, beta)
           
            # set v to the max(v, v')
            # sets the action part of the tuple as well
            if vPrime[0] > v[0]:
                v = (vPrime[0], action)
            # if v is greater than Beta,
            # it means that v is greater than MIN"s
            # best option on path to root, so we can prune
            # and stop exploring that branch of the tree.
            if v[0] > beta[0]:
                return v
            # check if we have found a new best alternative for MAX
            # update it if we have
            if v[0] > alpha[0]:
                alpha = (v[0], action)
            
        # return v, tuple with value and action
        return v
                
          
    def minValue(self, gameState, agentIndex, currentDepth, alpha, beta):
        # if the state is a terminal state:
        # return the state's utility
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), Directions.STOP)
        
        
        # retrieve the number of agents,
        # this includes all ghosts and Pacman
        
        numAgents = gameState.getNumAgents()
        # the next agent after every ghost except for the
        # last ghost is a MIN agent
        # for a ghost that isn't the one with the highest index
        if (agentIndex < numAgents-1):
            # initialize float to negative infinity
            v = (float("inf"), Directions.STOP)
            
            # get the legal actions the ghost can take
            legalActions = gameState.getLegalActions(agentIndex)
            
            # for each successor of state
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # call minValue with incremented index to get v'
                vPrime = self.minValue(successor, agentIndex+1, currentDepth, alpha, beta)
                
                # v = min(v, v')
                if vPrime[0] < v[0]:
                    v = (vPrime[0], action)
                # if v is less than alpha,
                # it means that v is less than MAX's best option on path to root
                # we can prune, don't need to keep exploring that branch.
                if v[0] < alpha[0]:
                    return v
                # check if we have found new best alternative for MIN
                # update it if we have
                if v[0] < beta[0]:
                    beta = (v[0], action)
                

            # return value, action tuple
            return v
            
        # the ghost with the highest index has Pacman
        # as the next agent
        elif (agentIndex == numAgents-1):
            # initialize v to infinity
            v = (float("inf"), Directions.STOP)
            
            # get the legal actions the ghost can take
            legalActions = gameState.getLegalActions(agentIndex)
            
            # for each successor of state
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # call maxValue function with Pacman's agentIndex 0
                # a single depth ply has Pacman and each ghost's actions
                # this means that this is where we increment the depth
                vPrime = self.maxValue(successor, 0, currentDepth+1, alpha, beta)
                
                # v = min(v, v')
                if vPrime[0] < v[0]:
                    v = (vPrime[0], action)
                
                # if v is less than alpha,
                # it means that v is less than MAX's best option on path to root
                # we can prune, don't need to keep exploring that branch.
                if v[0] < alpha[0]:
                    return v
                # check if we have found new best alternative for MIN
                # update it if we have
                if v[0] < beta[0]:
                    beta = (v[0], action)
                
            # return value, action tuple
            return v
        
        
            
            
    
    
    

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        # initialize agentIndex to 0 for root Pacman
        # initialize currentDepth to 0
        agentIndex = 0
        currentDepth = 0
        # call maxValue function for Pacman
        utility = self.maxValue(gameState, agentIndex, currentDepth)
        # utility is a tuple with value and action
        # return the action portion of the tuple
        return utility[1]
    
    def maxValue(self, gameState, agentIndex, currentDepth):
        # if the state is a terminal state:
        # return the state's utility
        if self.depth==currentDepth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), Directions.STOP)
    
        # we find what legal actions Pacman can take
        legalActions = gameState.getLegalActions(agentIndex)
        # initialize v to negative infinity
        v = (float("-inf"), Directions.STOP)
    
        # for each successor of state
        for action in legalActions:
            successor = gameState.generateSuccessor(agentIndex, action)
            # call the minValue function with incremented agentIndex
            vPrime = self.expValue(successor, agentIndex+1, currentDepth)
       
            # v = max(v, v')
            if vPrime[0] > v[0]:
                v = (vPrime[0], action)
            
        # return value, action tuple
        return v
    
    def expValue(self, gameState, agentIndex, currentDepth):
        """
        Instead of min nodes, we no longer assume that the ghosts
        play optimally.Now there are chance nodes
        with expected values.
        """
        # if the state is a terminal state:
        # return the state's utility
        if self.depth == currentDepth or gameState.isLose() or gameState.isWin():
            return (self.evaluationFunction(gameState), Directions.STOP)
    
        # retrieve the number of agents,
        # this includes all ghosts and Pacman
    
        numAgents = gameState.getNumAgents()
    
        # the next agent after every ghost except for the
        # last ghost is a EXP agent
        if (agentIndex < numAgents-1):
            # initialize v to 0
            v = (0.0, Directions.STOP)
            # get the legal actions the ghost can take
            legalActions = gameState.getLegalActions(agentIndex)
            # for each successor of state
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # assume you will be running against an adversary
                # who chooses amongst their legalActions uniformly at random
                numLegalActions = float(len(legalActions))
                # p = probablity(successor)
                # make sure we are using floats
                # because integer division in Python truncates
                p = 1.0/numLegalActions
                
                # call expected value function with incremented agentIndex
                vPrime = self.expValue(successor, agentIndex+1, currentDepth)
                
                # v += probability * value(successor)
                value = v[0]
                value += p * vPrime[0]
                # v is tuple with both value and action
                v = (value, action)
            
            return v
        
        # the next agent after the ghost with the highest index
        # is Pacman
        elif (agentIndex == numAgents-1):
            # initialize v to 0
            v = (0.0, Directions.STOP)
            
            # figure out what legal actions the ghost can take
            legalActions = gameState.getLegalActions(agentIndex)
            # for each successor of state
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # assume you will be running against an adversary
                # who chooses amongst their legalActions uniformly at random
                numLegalActions = float(len(legalActions))
                p = 1.0/numLegalActions
                # call the maxValue function with Pacman, agentIndex 0
                # 1 depth ply is Pacman and all ghosts.
                # increment the depth.
                vPrime = self.maxValue(successor, 0, currentDepth+1)
                # v += p * value(successor)
                value = v[0]
                value += p * vPrime[0]
                # v is a tuple with both value and action
                v = (value, action)
            return v
    

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
      I used a feature-based representation, describing states with
      a vector of features. Features are functions from states to
      real numbers that capture important properties of the state.
      I realized that since the ghost moves randomly and doesn't chase
      Pacman, the most important feature is eating the food pellets.
      The evaluation function also takes into account distance to
      capsules, number of capsules, distance to closest ghost, and
      whether a ghost is currently scared or active. I used feature
      representation to write a value function for a state using
      a few weights. In this case, I set the weights based on the relative
      importance of each feature and manually testing whether the weight
      improved the game performance.
    """
    "*** YOUR CODE HERE ***"
   
    
    # get Pacman's position
    pacmanPosition = currentGameState.getPacmanPosition()
    
    # the location of the food pellets are the most important thing
    # ghosts move randomly so we want Pacman to be optimistic
    newFood = currentGameState.getFood()
    
    # turn boolean grid of food positions into a list
    # this will be a list of only the positions
    # where there is food
    foodPositions = newFood.asList()
    """
    calculate the Manhattan Distance
    from Pacman's position to every element in the list of food positions
    I looked at this source to figure out how to do this with list comprehension
    https://www.kite.com/python/answers/how-to-apply-a-function-to-a-list-in-python
    """
    # calculate Manhattan distance between Pacman and all food pellets
    foodDistance = [util.manhattanDistance(element, pacmanPosition) for element in foodPositions]
    
    if foodDistance:
           closestFood = min(foodDistance)
       # if all food is gone, set closest food to 0
       # so it doesn't have an impact
    else:
        closestFood = 100
    
    # in addition to food pellets,
    # evaluation function needs to take into account ghosts
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    # use Manhattan distance between Pacman and ghost
    ghostPositions = currentGameState.getGhostPositions()
    ghostDistance = [util.manhattanDistance(element, pacmanPosition) for element in ghostPositions]
    
    # adjust Pacman's behavior to the closest ghost
    if ghostDistance:
        closestGhost = min(ghostDistance)
    else:
         closestGhost = 100
         
    # figure out whether the ghost is scared or active
    if max(newScaredTimes) > 0:
        scaredGhost = True
        activeGhost = False
    else:
        scaredGhost = False
        activeGhost = True
    # if the ghost is scared, Pacman should chase the ghost.
    if scaredGhost:
        scaredGhostScore = 1/closestGhost
    else:
        scaredGhostScore = 0
    # if the ghost is active, Pacman should stay away from the ghost.
    if activeGhost and closestGhost <= 1:
        activeGhostScore = -10
    else: activeGhostScore = 0
    
    # eating a capsule sets the scared timer
    # Pacman gets 200 points for eating a scared ghost.
    # calculate Manhattan Distance between Pacman and capsules.
    capsulePositions = currentGameState.getCapsules()
    capsuleDistance = [util.manhattanDistance(element, pacmanPosition) for element in capsulePositions]
    if capsuleDistance:
        closestCapsule = min(capsuleDistance)
    else:
        closestCapsule = 0.01
    
    # if Pacman is close the capsule
    # the capsule will impact his action's
    # otherwise, his focus will be on getting the food pellets
    if closestCapsule < 1:
        capsuleScore = 1
    else:
        capsuleScore = 0
    
    # give Pacman incentive to actually eat the capsule
    # rather than get stuck thrashing close to the capsule
    numCapsules = len(capsulePositions)
    if numCapsules == 0:
        numCapsulesScore = 50
    else:
        numCapsulesScore = 0
    currentScore = currentGameState.getScore()
    
    # linear evaluation function with manually set coefficients
    score = currentScore + (1/(closestFood ** 2)) + scaredGhostScore + activeGhostScore +  (10 * capsuleScore) + (10 * numCapsulesScore)
    
    
    return score

# Abbreviation
better = betterEvaluationFunction
