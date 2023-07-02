# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


        
        
    
def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]



    

def depthFirstSearch(problem: SearchProblem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
                                              # To write this code i followed exactly the algorithm of page 53 of Unit 2 and i .
    node=(problem.getStartState(),None,None,0,0) # A node is a tuple that contains state,parent,action,path_cost,depth just like page 19 of Unit 3 .
    explored=set() #explored will be a set in which we add a state every time the algorithm sees a new state , in order to avoid checking a state again. Every time we review a state we check if it is inside explored so that we know if the node is already explored.
    fringe=util.Stack() #fringe will be a stack which we will use in order to expand a node , everytime we remove a node we expand it , expanding means we add this node's children in the stack.
    fringe.push(node)  #We start by adding the initial node in the Stack.

    while not fringe.isEmpty(): # While fringe isn't empty we do the following.
        node=fringe.pop()  # At first we remove the only node the stack has (the fisrt) in order to review it later on.
        parent=node #The variable parent gets the content of the node , because we will use it later to create the children of a node.
       
                        
        if problem.isGoalState(node[0]): # If the state  of the node we check , is the goal state then we search for the previous actions that were took in order to get to that state. 
            list_of_actions=[] #This will be a list that contains the actions that were took in order to get to the goal state.
            iteration=node    #We assing the content of node to the variable iteration , to do the checking below.
        
            while iteration[1]!=None: #While the state that we check  isn't the starting state then go inside the while loop . iteration[1]==None means that the node we check has no parent , obviously that node is the starting node. 
           
            
             list_of_actions.append(iteration[2]) #We add the action to the list .
                
                
             iteration=iteration[1] #The variable iteration now 'points' to the parent of the node we checked,  and every time we go above the node we checked untill we arrive at the beggining.
            
                
            list_of_actions.reverse() #As we add actions , the first action that is added is the last action that got us to the goal state , after that we add the action before that e.t.c .So the last action that we add is the first so we just reverse the list to get the actions in order from start to finish.
            
                    
            return  list_of_actions    #Finally we return the list of actions.
                
        if node[0] not in explored:  #If the node we check isn't already checked the we do the following:
            explored.add(node[0])       # We add the state in the explored set.
            for x in problem.getSuccessors(node[0]): #We add the children of the node in the stack (fringe) .
                 
             
             child=[x[0],parent,x[1],x[2]+node[3],1]  #We create the child 
             iterator=child              #The variable iterator gets the content of the child .
             
             
             while iterator[1]!=None :  #This while loop calculates the path-cost.
               child[4]=child[4]+1
               iterator=iterator[1]
             variable=child[4]  #This variable holds the path-cost.
             child=(x[0],parent,x[1],x[2]+parent[3],variable) #The child spesifically has a state,a parent,the action that happened to get to that child,cost of the path untill the child,the depth (the number of nodes that took to get there)
             node=child #The variable node now gets the content of the child 
             fringe.push(node)     # We push the node in the stack.
               
           
    return False  #If nothing happens we return false
    

def breadthFirstSearch(problem: SearchProblem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
  
    node=(problem.getStartState(),None,None,0,0)   #We have the same algorithm here with the algorithm we had in dfs , but the only change is that we use a Queue.
                                                   #It is exactly the same node as in BFS with the only change of the queue , we do that in the following questions till q4 , we basically use the same code as in dfs and bfs but with small changes
    explored=set() 
    fringe=util.Queue() 
    fringe.push(node)  
    
    
    while not fringe.isEmpty():
        node=fringe.pop()
        parent=node
       
                       
        if problem.isGoalState(node[0]): 
            list_of_actions=[]
            iteration=node
            
            while iteration[1]!=None:
           
            
             list_of_actions.append(iteration[2])
                
                
             iteration=iteration[1]
               
                
            list_of_actions.reverse()
            
                    
            return  list_of_actions    
           
                
        if node[0] not in explored:
            explored.add(node[0])       
            for x in problem.getSuccessors(node[0]):
                 
             
             child=[x[0],parent,x[1],x[2]+node[3],1]
             iterator=child
             
             
             while iterator[1]!=None :
               child[4]=child[4]+1
               iterator=iterator[1]
             variable=child[4]
             child=(x[0],parent,x[1],x[2]+parent[3],variable)
             node=child
             fringe.push(node)     
               
           
    return False         
   

def uniformCostSearch(problem: SearchProblem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    node=(problem.getStartState(),None,None,0,0)  #We have the same algorithm here with the algorithm we had in dfs , but the only change is that we use a priority queue.
    
    explored=set() 
    fringe=util.PriorityQueue() 
    fringe.push(node,0)  

    while not fringe.isEmpty():
        node=fringe.pop()
        parent=node
       
                        
        if problem.isGoalState(node[0]): 
            list_of_actions=[]
            iteration=node
            
            while iteration[1]!=None:
           
            
             list_of_actions.append(iteration[2])
                
                
             iteration=iteration[1]
               
                
            list_of_actions.reverse()
            
                    
            return  list_of_actions    
                
        if node[0] not in explored:
            explored.add(node[0])       
            for x in problem.getSuccessors(node[0]):
                 
             
             child=[x[0],parent,x[1],x[2]+node[3],1]
             iterator=child
             
             
             while iterator[1]!=None :
               child[4]=child[4]+1
               iterator=iterator[1]
             variable=child[4]
             child=(x[0],parent,x[1],x[2]+parent[3],variable)
             node=child
             fringe.push(node,node[3])     #We push the node with its priority , here we push it with the cost of getting to that node . 
               
           
    return False  
    

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic):  #We have the same algorithm here with the algorithm we had in dfs , but the only change is that we use a priority queue.
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    node=(problem.getStartState(),None,None,0,0)
    
    explored=set() 
    fringe=util.PriorityQueue() 
    fringe.push(node,0)  

    while not fringe.isEmpty():
        node=fringe.pop()
        parent=node
       
                        
        if problem.isGoalState(node[0]): 
            list_of_actions=[]
            iteration=node
            
            while iteration[1]!=None:
           
            
             list_of_actions.append(iteration[2])
                
                
             iteration=iteration[1]
               
                
            list_of_actions.reverse()
            
                    
            return  list_of_actions    
                
        if node[0] not in explored:
            explored.add(node[0])       
            for x in problem.getSuccessors(node[0]):
                 
             
             child=[x[0],parent,x[1],x[2]+node[3],1]
             iterator=child
             
             
             while iterator[1]!=None :
               child[4]=child[4]+1
               iterator=iterator[1]
             variable=child[4]
             child=(x[0],parent,x[1],x[2]+parent[3],variable)
             node=child
             fringe.push(node,node[3]+heuristic(node[0],problem))    # A* star's evaluation function is equal to cost of path + cost of heuristic , so we push in the fringe the node and  the cost of the evaluation function . We push it into a priority queue  so we push the node and its priority , which is the evaluation function .
               
           
    return False  
   


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
