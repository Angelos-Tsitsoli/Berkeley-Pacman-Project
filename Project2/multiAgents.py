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
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates] #exei themaaa edooooo
        "*** YOUR CODE HERE ***"
        #print("succesorgamestate")
        #print(successorGameState)  #deixnei to game me dieseis
        #print("newpos")
        #print(newPos) #px (1,2)
        #print("newfood")
        #print(newFood) #printarei FFFFF
        #print("newGhostStates")
        #print(newGhostStates) #[<game.AgentState object at 0x0000019D5F27F978>]
        #print("newScaredTimes")
        #print(newScaredTimes) #[0]
        
        #successorGameState.getGhostStates      
        the_score=0 #the final score , which we will return .
        if successorGameState.isWin():  #If the state is the winning state.
            return 1000000 #Just a very big number for the score
        
        thesi=currentGameState.getPacmanPosition() #Pacman's current position
       
        
        #distance from ghost
        if thesi in currentGameState.getGhostPositions() : #if current position has a ghost
            return -1000000
        
        if newPos in currentGameState.getGhostPositions():  #if successor has a ghost 
            return -500000
        
        if newPos in  currentGameState.getCapsules(): #If at the new position is eating a capsule.
            return 10000
        
        p_dis_from_dots=[] #array for the distances of pacman from foods of the new state 
        p_dis_from_g=[]  #array for the distances of pacmsn from ghosts
        p_dis_from_curdots=[] #array for the distance of pacman from dots at the current state
              
                 
        ##ghost min bale succ kai curr     
        
        the_score = the_score + successorGameState.getScore() - currentGameState.getScore() #the difference between the current score and a future score .If the future is bigger than the previous score we add points if its not the we subtract points
        #manhattan distance to closest food from successor
        for x in newFood.asList():#manhattan distance for dots from the new position 
            p_dis_from_dots.append(util.manhattanDistance(newPos,x))
            
        for x in currentGameState.getFood().asList():#manhattan to closest food from current state 
            p_dis_from_curdots.append(util.manhattanDistance(currentGameState.getPacmanPosition(),x))
   
    
        #manhattan distance to closest ghost from successor
        for x in successorGameState.getGhostPositions(): #The distance between the new position and ghosts
            p_dis_from_g.append(util.manhattanDistance(newPos,x))   
            
        #for x in currentGameState.getGhostPositions(): 
       #     p_dis_from_gcur.append(util.manhattanDistance(newPos,x))     
        
        
        if min(p_dis_from_dots) < min(p_dis_from_g): #if min distance from succ to food is less than min dist from suc to ghost
            the_score=the_score + 100
            
        else : the_score=the_score - 500
        
        
        #distance from ghost
        if thesi in currentGameState.getGhostPositions() : #if current position has a ghost
            return -1000000
        
        if newPos in currentGameState.getGhostPositions():  #if successor has a ghost 
            return -500000
        
        if newPos in  currentGameState.getCapsules(): #if new position is in one of the capsules
            return 10000
        
        if len(currentGameState.getFood().asList()) > len(newFood.asList()):   #if the dots before were more than after
            the_score=the_score+500
        else:
            the_score=the_score-100
        
        if len(currentGameState.getCapsules()) > len(successorGameState.getCapsules()):   #if the capsules then lesser than succ
            the_score=the_score + 2000   
        
        if min(p_dis_from_dots)<min(p_dis_from_curdots):    #if min dist from dot is less than the min distan after
            the_score=the_score+300
        else:
            the_score=the_score-250
    
        return the_score

def scoreEvaluationFunction(currentGameState: GameState):
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

    def MinValue(self,gameState,depth,agentIndex): 
         v=float('Inf')
         the_number=gameState.getNumAgents()-1 #the number of ghosts
         if gameState.isWin():                           #if that gamestate is a state of winning or a loss then return the value of the evaluation function.
             return self.evaluationFunction(gameState)
         
         if gameState.isLose():
             return self.evaluationFunction(gameState)
         
         for a in gameState.getLegalActions(agentIndex):  #as we iterate through the actions that we can take with the specific ghost  we do the following
             the_suc=gameState.generateSuccessor(agentIndex,a)  #we store the successor in the variable the_suc.
             if agentIndex==the_number:       #if we check the last ghost of the ghosts then we have to call the MaxValue function because its pacmans turn
                 v=min(v,self.MaxValue(the_suc,depth+1)) #we update the v from the  minimun between the v number we check that moment  and the value that comes from pacmans turn at the new depth.
             #else:#allakse to me if 
             if agentIndex!=the_number:          #if we have other ghosts to check then we continue by calling the MinValue function and updating the variable v with the minimum value between v and the value that is returned from the MinValue function that was called for another ghost
                 v=min(v,self.MinValue(the_suc,depth,agentIndex+1))   #We give as an AgentIndex the next ghost in order to check the values there also.
         return v

    def MaxValue(self,gameState,depth):
        #the_depth=depth+1
        v=float('-Inf')
        
        if depth==self.depth:             #We check if the depth is at the limit 
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        
        for a in gameState.getLegalActions(0):    #We give zero because we want to see the legal actions of the pacman which agent number is zero
            the_suc=gameState.generateSuccessor(0,a) #We store the successors of the pacman.
            v=max(v,self.MinValue(the_suc,depth,1))  #The parameters #We store at the v variable the maximum between the current v and the value that the ghosts wil return .
        
        return v
        

    def getAction(self, gameState: GameState):
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
       
    
            
        action=None              #Thats the start of the process .Its like we are at the top of the game tree and we are at the root node and its pacmans turn.
        root_state_actions=gameState.getLegalActions(0) #The legal actions for pacman
        the_depth=0            #At first the depth is zero
        score=float('-Inf')          #The score 
        for x in root_state_actions:
            the_suc=gameState.generateSuccessor(0,x) #we iterate through the successors of the root 
            i=self.MinValue(the_suc,the_depth,1) #we store to the variable i what the function MINValue returns 
            if i>score:       #everytime we that the i variable has a bigger value than the variable score then we store to score the value and we keep the variable x (which has the action ) at the action variable.
                score=i         #basically we want to return the action which is responsible for the biggest value of the variable score.
                action=x
        return action #Then we return the variable action 
            
      

class AlphaBetaAgent(MultiAgentSearchAgent):    #This is the same algorithm as minimax with a small addition.
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """
    def MaxValue(self,gameState,depth,a,b):     #Basically  at the MaxValue function , we check if the v variable is bigger than the b variable (which has the smallest value so far) and if its true we dont have to do further investigations
        #the_depth=depth+1                          #We also update the a variable 
        v=float('-Inf')
                                  
        if depth==self.depth:
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        
        for x in gameState.getLegalActions(0):
            the_suc=gameState.generateSuccessor(0,x)  #Just like the code we used in class
            v=max(v,self.MinValue(the_suc,depth,1,a,b))
            if v>b:
                return v
            a=max(a,v)
        return v
    
    def MinValue(self,gameState,depth,agentIndex,a,b):#At the MinValue function , we check if the v variable is smaller than the a variable (which has the biggest value so far) and if its true we dont have to do further investigations
        v=float('Inf')                                   #We also update the b variable .
        the_number=gameState.getNumAgents()-1 #only the ghosts
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        
        for x in gameState.getLegalActions(agentIndex):
            the_suc=gameState.generateSuccessor(agentIndex,x)
            if agentIndex==the_number:
                v=min(v,self.MaxValue(the_suc,depth+1,a,b))
                
                if v<a:
                 return v
                b=min(b,v)
                
            if agentIndex!=the_number:
                v=min(v,self.MinValue(the_suc,depth,agentIndex+1,a,b))
                if v<a:
                    return v
                b=min(b,v)      
        return v
   
    
    
    
    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        action=None
        root_state_actions=gameState.getLegalActions(0) 
        the_depth=0               
        score=float('-Inf')      
        a=float('-Inf')     #We initialize the a and b variables
        b=float('Inf')
        for x in root_state_actions:
            the_suc=gameState.generateSuccessor(0,x) 
            i=self.MinValue(the_suc,the_depth,1,a,b)
            if i>score:
                score=i
                action=x
            if i>b:     #We also check here as in MaxValue  if i is bigger than the variable b then we dont have to check anymore,we just return the action.
                return action
            a=max(a,i)
        return action
    #    util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):    #We also use the minimax algorithm here but the difference here is that every time in order to calculate the v value as before at the MINValue function , here in the chance value function we in order to calulate value v we have to find the average .In order to find the average everytime we divide the value of v with the number of legal actions that can happen.
    """
      Your expectimax agent (question 4)
    """
    def ChanceValue(self,gameState,depth,agentIndex):
         v=0#float('Inf')
         the_number=gameState.getNumAgents()-1 #only the ghosts
         if gameState.isWin():
             return self.evaluationFunction(gameState)
         
         if gameState.isLose():
             return self.evaluationFunction(gameState)
         n=len(gameState.getLegalActions(agentIndex))  
         for a in gameState.getLegalActions(agentIndex):
             the_suc=gameState.generateSuccessor(agentIndex,a)
             if agentIndex==the_number:
                 v+=(1/n)*self.MaxValue(the_suc,depth+1)
             if  agentIndex!=the_number:
                 v+=(1/n)*self.ChanceValue(the_suc,depth,agentIndex+1)
         return v

    def MaxValue(self,gameState,depth):
        #the_depth=depth+1
        v=float('-Inf')
        
        if depth==self.depth:
            return self.evaluationFunction(gameState)
        if gameState.isWin():
            return self.evaluationFunction(gameState)
        if gameState.isLose():
            return self.evaluationFunction(gameState)
        
        for a in gameState.getLegalActions(0):
            the_suc=gameState.generateSuccessor(0,a)  #Just like the code we used in class
            v=max(v,self.ChanceValue(the_suc,depth,1))
        
        return v
        
    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        action=None
        root_state_actions=gameState.getLegalActions(0) 
        the_depth=0
        score=float('-Inf')
        for x in root_state_actions:
            the_suc=gameState.generateSuccessor(0,x)
            i=self.ChanceValue(the_suc,the_depth,1)
            if i>score:
                score=i
                action=x
        return action
    
    
    
        #util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newGhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    
    #In this function aas in the first if we get in a good position in order to reach the goal state we give a good score else the score is bad.
    
    if currentGameState.isWin():
        return float('Inf')
    

    nF = len(currentGameState.getFood().asList())#The number of food left
    foods=[]   #the distances of pacman from food
    ag=[] #the distances of pacman from ghosts
    the_sum=sum(ScaredTimes) #the sum of the scaredtimes tha we'll need after in order to see if there are any scared ghosts
    
      
    
    if currentGameState.getFood().asList():               #if we have food remaining then we find the manhatan distances from them and we take the minimum of them
     for iterator in currentGameState.getFood().asList():
        i=util.manhattanDistance(currentGameState.getPacmanPosition(),iterator)
        foods.append(i)
     the_min=min(foods)  
      
    if not currentGameState.getFood().asList(): #if not we return the worst score 
        the_min=float('Inf')
    
    if currentGameState.getGhostStates():  #if we have ghosts remaining then we find the manhatan distances from them and we take the minimum of them
        for iterator in currentGameState.getGhostStates():
         i=util.manhattanDistance(currentGameState.getPacmanPosition(),iterator.getPosition())
         ag.append(i)
        the_min2=min(ag)
        
    if not currentGameState.getGhostStates(): #if not thats great we return the best score
        the_min2=float('Inf') 
      
        
    if the_min2==0: #if we are very close to a ghosts then return the worst score
        return -float('inf')    
     
     
    if the_sum >0: #If we hac\ve scared ghosts then do the following 
        return currentGameState.getScore() + 1/the_min + 1/the_min2 +  nF
     
    else: #if not then that means we have ghosts remaining
        return currentGameState.getScore() + 1/the_min -1/the_min2  +  nF
    


# Abbreviation3
better = betterEvaluationFunction
