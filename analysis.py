# analysis.py
# -----------
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


######################
# ANALYSIS QUESTIONS #
######################

# Set the given parameters to obtain the specified policies through
# value iteration.

def question2():
    #answerDiscount = 0.9
    answerDiscount = 0.9
    """
    The agent doesn't try to cross the bridge
    because the noise makes it so crossing the
    bridge carries a pretty high risk of falling
    into the negative pits on the side of the bridge.
    If there isn't noise, the agent will try to cross
    """
    answerNoise = 0.0
    #answerNoise = 0.2
    return answerDiscount, answerNoise

def question3a():
    """
    setting of the parameter values for each part
    should have the property that if agent followed the
    optimal policy without being subject to any noise,
    it would exhibit the given behavior.
    Prefer the close exit (+1), risking the cliff (-10)
    """
    # discount incentivizes agent to choose
    # closer reward over the reward further away
    answerDiscount = 0.1
    answerNoise = 0.0
    # negative living reward causes agent to risk the cliff
    # instead of taking the longer, safer route
    answerLivingReward = -0.4
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3b():
    """
    Prefer the close exit (+1), but avoiding the cliff (-10)
    """
    # discount incentivizes agent to choose
    # closer reward over reward further away
    answerDiscount = 0.1
    # without noise, agent keeps heading east
    # toward higher reward, instead of going
    # south to lower reward
    answerNoise = 0.1
    # don't want agent to be penalized for staying alive
    # this causes agent to take longer path
    # instead of risking the cliff
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3c():
    """
    Prefer the distant exit (+10), risking the cliff (-10)
    """
    # has effect of no Discount
    # incentivizes agent to go for further away reward
    answerDiscount = 1.0
    # no noise
    answerNoise = 0.0
    # living reward incentivizes risking the cliff
    # instead of going slow way around
    answerLivingReward = -0.4
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3d():
    """
    Prefer the distant exit (+10), avoiding the cliff (-10)
    """
    # has effect of no Discount
    # incentivizes agent to go for further away reward
    answerDiscount = 1.0
    # with smaller noise, agent went to lower reward
    # with more noise, agent goes to higher reward
    answerNoise = 0.5
    # there isn't a negative living reward so
    # agent avoids the cliff and takes the safer path
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question3e():
    """
    Avoid both exits and the cliff (so an episode should never terminate)
    """
    # discount doesn't matter
    # could be 0.0 or 1.0
    answerDiscount = 0.0
    # super noisy
    answerNoise = 1.0
    answerLivingReward = 0.0
    return answerDiscount, answerNoise, answerLivingReward
    # If not possible, return 'NOT POSSIBLE'

def question8():
    answerEpsilon = None
    answerLearningRate = None
    #return answerEpsilon, answerLearningRate
    # If not possible, return 'NOT POSSIBLE'
    """
    Is there an epsilon and a learning rate for which it is highly likely (greater
    than 99%) that the optimal policy will be learned after 50 iterations?
    There is not an epsilon and a learning rate for which there is a greater than
    99% chance that the optimal policy will be learned after only 50 iterations.
    """
    return 'NOT POSSIBLE'

if __name__ == '__main__':
    print('Answers to analysis questions:')
    import analysis
    for q in [q for q in dir(analysis) if q.startswith('question')]:
        response = getattr(analysis, q)()
        print('  Question %s:\t%s' % (q, str(response)))
