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


import random

import util
from game import Agent, Directions
from util import manhattanDistance



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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        x= func1(successorGameState)
        return x
        #return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

def func1(state):
    newPos = state.getPacmanPosition()
    newFood = state.getFood()
    food_list = newFood.asList() #take new food as list
    min_food_distance = -1   #default minimum distance to food
    for food in food_list:
            distance = util.manhattanDistance(newPos, food)  #evaluate the manhattan distance 
            if min_food_distance >= distance or min_food_distance == -1:
                min_food_distance = distance

    distance_ghost = 1    
    ghost_proximity = 0
    ghost_postion =state.getGhostPositions()
    for ghost_state in ghost_postion:
            distance = util.manhattanDistance(newPos, ghost_state) #calculate distance from ghost
            distance_ghost += distance
            if distance <= 1:
                ghost_proximity += 1
    buffer = 1 / min_food_distance
    buffer1 = 1 / distance_ghost
    buffer2 = buffer + buffer1 - ghost_proximity
    return state.getScore() + buffer2
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        maximum = float("-inf")
        default_inf = float("-inf")
        action = Directions.EAST
        def minimax(current_agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
               # return the utility in when the needed depth is reached or if the game is either  won or lost. 
                return self.evaluationFunction(gameState)
            if current_agent == 0: 
                return max(minimax(1, depth, gameState.generateSuccessor(current_agent, new_state)) 
                           for new_state in gameState.getLegalActions(current_agent))
            else: 
                new_agent = current_agent + 1 
                if gameState.getNumAgents() == new_agent:
                    new_agent = 0
                if new_agent == 0:
                   depth += 1
                return min(minimax(new_agent, depth, gameState.generateSuccessor(current_agent, new_state))
                           for new_state in gameState.getLegalActions(current_agent))
        
        for agent_state in gameState.getLegalActions(0):
            utility_func = minimax(1, 0, gameState.generateSuccessor(0, agent_state))
            if utility_func > maximum or maximum == default_inf:
                maximum = utility_func
                action = agent_state

        return action

        util.raiseNotDefined()


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action = Directions.EAST
        utility_func = float("-inf")
        alpha =  float("-inf")
        beta = float("inf")
        def alphabeta(current_agent, depth, game_state, alpha, beta):
            if game_state.isLose() or game_state.isWin() or depth == self.depth:  
                # return the utility in case where the defined depth is reached or  when the game is won/lost.
                return self.evaluationFunction(game_state)

            if current_agent == 0:  
                return alphabeta_max(current_agent, depth, game_state, alpha, beta)
            else:  
                return alphabeta_min(current_agent, depth, game_state, alpha, beta)

        def alphabeta_max(current_agent, depth, game_state, a, b):  
            v = float("-inf")
            for new_state in game_state.getLegalActions(current_agent):
                v = max(v, alphabeta(1, depth, game_state.generateSuccessor(current_agent, new_state), a, b))
                if v > b:
                    return v
                a = max(a, v)
            return v

        def alphabeta_min(current_agent, depth, game_state, a, b):  
            
            v = float("inf")
            next_agent = current_agent + 1  # calculate the next agent and increase depth.
            if game_state.getNumAgents() == next_agent:
                next_agent = 0
            if next_agent == 0:
                depth += 1

            for new_state in game_state.getLegalActions(current_agent):
                v = min(v, alphabeta(next_agent, depth, game_state.generateSuccessor(current_agent, new_state), a, b))
                if v < a:
                    return v
                b = min(b, v)
            return v
        
        for agent_state in gameState.getLegalActions(0):
            ghost = alphabeta(1, 0, gameState.generateSuccessor(0, agent_state), alpha, beta)
            if ghost > utility_func:
                utility_func = ghost
                action = agent_state
            if utility_func > beta:
                return utility_func
            alpha = max(alpha, utility_func)

        return action

        util.raiseNotDefined()


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
        maximum = float("-inf")
        default_inf = float("-inf") # infinity
        action = Directions.EAST
        def expectimax(current_agent, depth, gameState):
            if gameState.isLose() or gameState.isWin() or depth == self.depth:  
                return self.evaluationFunction(gameState)
            if current_agent == 0: 
                return max(expectimax(1, depth, gameState.generateSuccessor(current_agent, new_state)) for new_state in gameState.getLegalActions(current_agent))
            else: 
                new_parameter = current_agent + 1  
                if gameState.getNumAgents() == new_parameter:
                    new_parameter = 0
                if new_parameter == 0:
                    depth += 1
                return sum(expectimax(new_parameter, depth, gameState.generateSuccessor(current_agent, new_state)) 
                           for new_state in gameState.getLegalActions(current_agent)) / (len(gameState.getLegalActions(current_agent)))

        
        for agent_state in gameState.getLegalActions(0):
            utility = expectimax(1, 0, gameState.generateSuccessor(0, agent_state))
            if utility > maximum or maximum == default_inf:
                maximum = utility
                action = agent_state

        return action

        util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    y =func1(currentGameState) 
    return y
    util.raiseNotDefined()


# Abbreviation
better = betterEvaluationFunction
