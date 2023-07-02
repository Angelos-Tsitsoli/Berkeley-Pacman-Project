# logicPlan.py
# ------------
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
In logicPlan.py, you will implement logic planning methods which are called by
Pacman agents (in logicAgents.py).
"""

from typing import Dict, List, Tuple, Callable, Generator, Any
import util
import sys
import logic
import game

from logic import conjoin, disjoin
from logic import PropSymbolExpr, Expr, to_cnf, pycoSAT, parseExpr, pl_true

import itertools
import copy

pacman_str = 'P'
food_str = 'FOOD'
wall_str = 'WALL'
pacman_wall_str = pacman_str + wall_str
ghost_pos_str = 'G'
ghost_east_str = 'GE'
pacman_alive_str = 'PA'
DIRECTIONS = ['North', 'South', 'East', 'West']
blocked_str_map = dict([(direction, (direction + "_blocked").upper()) for direction in DIRECTIONS])
geq_num_adj_wall_str_map = dict([(num, "GEQ_{}_adj_walls".format(num)) for num in range(1, 4)])
DIR_TO_DXDY_MAP = {'North':(0, 1), 'South':(0, -1), 'East':(1, 0), 'West':(-1, 0)}


#______________________________________________________________________________
# QUESTION 1

def sentence1() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    A or B
    (not A) if and only if ((not B) or C)
    (not A) or (not B) or C
    """
    "*** BEGIN YOUR CODE HERE ***"
    list1=[]  
    list2=[]
    A=Expr('A') #Creation of expressions
    B=Expr('B') 
    C=Expr('C') 
    AorB=A | B  #A or B
    negBorC= ~ B | C #((not B) or C)
    negA=~ A # not A
    negB=~ B #not B
    AequivalenceBorC=negA % negBorC #(not A) if and only if ((not B) or C)
    
    list1.append(negA) #We put them in list1 so that at the end we can conjoin them
    list1.append(negB)
    list1.append(C)
    
    list2.append(AorB) ##We put them in list2 so that at the end we can conjoin them
    list2.append(AequivalenceBorC)
    list2=conjoin(list2)
    
    temp=disjoin(list1) #The or in :(not A) "or" (not B) "or" C
    
    return conjoin([list2,temp]) #Return of the conjuction of all
    #util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def sentence2() -> Expr:
    """Returns a Expr instance that encodes that the following expressions are all true.
    
    C if and only if (B or D)
    A implies ((not B) and (not D))
    (not (B and (not C))) implies A
    (not D) implies C
    """
    "*** BEGIN YOUR CODE HERE ***"
    list1=[] #A list to hold the final conjuction
    A=Expr('A') #Creation of the expressions
    B=Expr('B')
    C=Expr('C')
    D=Expr('D')
    negD=~ D  #Creation of sentences
    negC=~ C #not C
    
    BorD=B | D  #B or D
    CequivalanceBorD=C % BorD  #C if and only if (B or D)
    
    negBandnegD=~B & ~D #
    AimplicationnegBandnegD= A >> negBandnegD #A implies ((not B) and (not D))
    
    BandnegC=B & negC  #B and not C
    negBandnegC=~BandnegC#(not (B and (not C)))
    
    negBandnegCimplicationA=negBandnegC >> A #(not (B and (not C))) implies A
    
    negDimplicationC=negD >> C #(not D) implies C)
    
    list1.append(CequivalanceBorD)
    list1.append(AimplicationnegBandnegD)
    list1.append(negBandnegCimplicationA)
    list1.append(negDimplicationC)
    list1=conjoin(list1)  #We conjoin the sentences in order to return them 
    return list1 
    
    #util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"


def sentence3() -> Expr:
    """Using the symbols PacmanAlive_1 PacmanAlive_0, PacmanBorn_0, and PacmanKilled_0,
    created using the PropSymbolExpr constructor, return a PropSymbolExpr
    instance that encodes the following English sentences (in this order):

    Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
    not killed at time 0 or it was not alive at time 0 and it was born at time 0.

    Pacman cannot both be alive at time 0 and be born at time 0.

    Pacman is born at time 0.
    (Project update: for this question only, [0] and _t are both acceptable.)
    """
    "*** BEGIN YOUR CODE HERE ***"
    #util.raiseNotDefined()
    list1=[] #A list to hold all the sentences and to  conjoin them all together in order to return them 
    
    str1=PropSymbolExpr('PacmanAlive', 0)  #Pacman is alive at time 0
    
    str2=PropSymbolExpr('PacmanAlive', 1)  #Pacman is alive at time 1
    
    str3=PropSymbolExpr('PacmanBorn', 0)   #Pacman is born at time 0
    
    str4=PropSymbolExpr('PacmanKilled', 0) #Pacman is killed at time 0
    
    negstr4=~str4 #The negation of the fourth sentence
    negstr1=~str1 #The negation of the first sentence
    
    str1andnegstr4ornegstr1andstr3=str1 & negstr4 | negstr1 & str3 #Pacman was alive at time 0 and it was
                                                                   #not killed at time 0 or it was not alive at time 0 and it was born at time 0
    
    equivalance=str2%str1andnegstr4ornegstr1andstr3 #Pacman is alive at time 1 if and only if Pacman was alive at time 0 and it was
                                                    #not killed at time 0 or it was not alive at time 0 and it was born at time 0
    
    negstr1andnegstr3=~ (str1 & str3) #Pacman cannot both be alive at time 0 and be born at time 0.
    
    list1.append(equivalance)  #Inserting the sentences
    list1.append(negstr1andnegstr3)
    list1.append(str3)
    list1=conjoin(list1)
    
    return list1

    
    "*** END YOUR CODE HERE ***"

def findModel(sentence: Expr) -> Dict[Expr, bool]:
    """Given a propositional logic sentence (i.e. a Expr instance), returns a satisfying
    model if one exists. Otherwise, returns False.
    """   
                              
    cnf_sentence = to_cnf(sentence)# Convert to cnf
    
    #>>> findModel(sentence1())
    #((A | B) & (~A <=> (~B | C)) & (~A | ~B | C))
    #{A: False, B: True, C: True}
    
    #>>> findModel(sentence2())
    #((C <=> (B | D)) & (A >> (~B & ~D)) & (~(B & ~C) >> A) & (~D >> C))
    #False
    
    #>>> findModel(sentence3())
    #((PacmanAlive[1] <=> ((PacmanAlive[0] & ~PacmanKilled[0]) | (~PacmanAlive[0] & PacmanBorn[0]))) & ~(PacmanAlive[0] & PacmanBorn[0]) & PacmanBorn[0])
    #{PacmanAlive[0]: False, PacmanKilled[0]: False, PacmanAlive[1]: True, PacmanBorn[0]: True}
   
     
    return pycoSAT(cnf_sentence) #Return a satisfying model

def findModelCheck() -> Dict[Any, bool]:
    """Returns the result of findModel(Expr('a')) if lower cased expressions were allowed.
    You should not use findModel or Expr in this method.
    This can be solved with a one-line return statement.
    """
    class dummyClass:
        """dummy('A') has representation A, unlike a string 'A' that has repr 'A'.
        Of note: Expr('Name') has representation Name, not 'Name'.
        """
        def __init__(self, variable_name: str = 'A'):
            self.variable_name = variable_name
        
        def __repr__(self):  
            return self.variable_name
    "*** BEGIN YOUR CODE HERE ***"

    dictionary={dummyClass('a'):True} #We have to return a dictionary as findmodel does so the variable "dictionary" will be what we will return .
                                       #By using the dummyclass we manage to get the represantation of a variable as it is and not like a string . For example by using the class dummyclass the variable A on the expresion that we want will be represanted as A and not as "A" (like a string).This happens because of the function repr that the dummyclass has .Specifically this function does not represent a string with the symbols "" instead it represents the string as it is .
    
    return dictionary
    
    
    
    
    "*** END YOUR CODE HERE ***"

def entails(premise: Expr, conclusion: Expr) -> bool:
    """Returns True if the premise entails the conclusion and False otherwise.
    """
    "*** BEGIN YOUR CODE HERE ***"
    #util.raiseNotDefined()
    final=[]  #This list holds the sentences that will be conjoined and will be given to findmodel to see if theres a satisfying model
    final.append(premise)
    final.append(~conclusion)  #In order for a premise to entail a conclusion . We must prove that there is not a model that satisfies the expression premise ^ ~conclusion (symbol '^' means logical "and" , symbol "~" means negative ) , by proving that we know for sure that the expression premise ^ conclusion is has a model
    final=conjoin(final)
    if findModel(final)==False: #If findmodel returns a satisfying model then we return true else we return false 
        return True
    else :
        return False
    
    "*** END YOUR CODE HERE ***"

def plTrueInverse(assignments: Dict[Expr, bool], inverse_statement: Expr) -> bool:
    """Returns True if the (not inverse_statement) is True given assignments and False otherwise.
    pl_true may be useful here; see logic.py for its description.
    """
    "*** BEGIN YOUR CODE HERE ***"
    #util.raiseNotDefined()
    
    negativeinvers=~inverse_statement   
    if pl_true(negativeinvers,assignments)==True : #Returns True if the pl_true function (not inverse_statement) is True given assignments and False otherwise
        return True
    else :
        return False
    
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 2

def atLeastOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals (i.e. in the form A or ~A), return a single 
    Expr instance in CNF (conjunctive normal form) that represents the logic 
    that at least one of the literals  ist is true.
    >>> A = PropSymbolExpr('A');
    >>> B = PropSymbolExpr('B');
    >>> symbols = [A, B]
    >>> atleast1 = atLeastOne(symbols)
    >>> model1 = {A:False, B:False}
    >>> print(pl_true(atleast1,model1))
    False
    >>> model2 = {A:False, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    >>> model3 = {A:True, B:True}
    >>> print(pl_true(atleast1,model2))
    True
    """
    "*** BEGIN YOUR CODE HERE ***"
    #util.raiseNotDefined()
    
    the_return=disjoin(literals)
    return the_return #At least one in order to be true , so we must have or between the expression in order for that to happen to we just disjoin them and return the result
    
    
    "*** END YOUR CODE HERE ***"


def atMostOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form) that represents the logic that at most one of 
    the expressions in the list is true.
    itertools.combinations may be useful here.
    """
    "*** BEGIN YOUR CODE HERE ***"
    #util.raiseNotDefined()
    lit=[]
    for x in literals :   #We only want one expression to be true so firstly we take the negative expression of every literal that is given 
        lit.append(~x)
    
    clauses=list(itertools.combinations(lit,2)) #We use itertools.combinations                                           
    lit2=[]
    final=[]                                       
    for x in clauses: #We take from every clause its expressions and we disjoin them (for every clause) then we conjoin all of them and return the result.
        for y in x:  
            lit2.append(y)   #This for loop is repeating as much as the expressions of the clause are .
        final.append(disjoin(lit2))
        lit2.clear()                   #Every time we get the expressions of a clause then we clear the list for the next clause              
    
    the_return=conjoin(final) # We conjoin the result to final and then we return the result 
    return the_return
    "*** END YOUR CODE HERE ***"


def exactlyOne(literals: List[Expr]) -> Expr:
    """
    Given a list of Expr literals, return a single Expr instance in 
    CNF (conjunctive normal form)that represents the logic that exactly one of 
    the expressions in the list is true.
    """
    "*** BEGIN YOUR CODE HERE ***"
    #util.raiseNotDefined()
    lit=[]
    for x in literals :
        lit.append(~x)
    clauses=list(itertools.combinations(lit,2)) 
    lit2=[]
    final=[]                                       
    for x in clauses: #We take from every clause its expressions and we disjoin them (for every clause) then we conjoin all of them and return the result.
        for y in x:   #This for loop is repeating as much as the expressions of the clause are .
            lit2.append(y)
        final.append(atLeastOne(lit2))
        lit2.clear() #Every time we get the expressions of a clause then we clear the list for the next clause 
    lit2=disjoin(literals)  
    final.append(lit2) 
    the_return=conjoin(final)
    return the_return 
    
        
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 3

def pacmanSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]=None) -> Expr:
    """
    Successor state axiom for state (x,y,t) (from t-1), given the board (as a 
    grid representing the wall locations).
    Current <==> (previous position at time t-1) & (took action to move to x, y)
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    """
    now, last = time, time - 1
    possible_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t
    # the if statements give a small performance boost and are required for q4 and q5 correctness
    if walls_grid[x][y+1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        possible_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not possible_causes:
        return None
    
    "*** BEGIN YOUR CODE HERE ***"
    
    #util.raiseNotDefined()
      
    #Current <==> (previous position at time t-1) & (took action to move to x, y)
    return PropSymbolExpr(pacman_str, x, y, time=now) % disjoin(possible_causes)
    
    
    "*** END YOUR CODE HERE ***"


def SLAMSuccessorAxiomSingle(x: int, y: int, time: int, walls_grid: List[List[bool]]) -> Expr:
    """
    Similar to `pacmanSuccessorStateAxioms` but accounts for illegal actions
    where the pacman might not move timestep to timestep.
    Available actions are ['North', 'East', 'South', 'West']
    """
    now, last = time, time - 1
    moved_causes: List[Expr] = [] # enumerate all possible causes for P[x,y]_t, assuming moved to having moved
    if walls_grid[x][y+1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y+1, time=last)
                            & PropSymbolExpr('South', time=last))
    if walls_grid[x][y-1] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x, y-1, time=last) 
                            & PropSymbolExpr('North', time=last))
    if walls_grid[x+1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x+1, y, time=last) 
                            & PropSymbolExpr('West', time=last))
    if walls_grid[x-1][y] != 1:
        moved_causes.append( PropSymbolExpr(pacman_str, x-1, y, time=last) 
                            & PropSymbolExpr('East', time=last))
    if not moved_causes:
        return None

    moved_causes_sent: Expr = conjoin([~PropSymbolExpr(pacman_str, x, y, time=last) , ~PropSymbolExpr(wall_str, x, y), disjoin(moved_causes)])

    failed_move_causes: List[Expr] = [] # using merged variables, improves speed significantly
    auxilary_expression_definitions: List[Expr] = [] 
    for direction in DIRECTIONS:
        dx, dy = DIR_TO_DXDY_MAP[direction]
        wall_dir_clause = PropSymbolExpr(wall_str, x + dx, y + dy) & PropSymbolExpr(direction, time=last)
        wall_dir_combined_literal = PropSymbolExpr(wall_str + direction, x + dx, y + dy, time=last)
        failed_move_causes.append(wall_dir_combined_literal)
        auxilary_expression_definitions.append(wall_dir_combined_literal % wall_dir_clause)

    failed_move_causes_sent: Expr = conjoin([
        PropSymbolExpr(pacman_str, x, y, time=last),
        disjoin(failed_move_causes)])

     
    return conjoin([PropSymbolExpr(pacman_str, x, y, time=now) % disjoin([moved_causes_sent, failed_move_causes_sent])] + auxilary_expression_definitions)


def pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
    """
    Given:
        t: timestep
        all_coords: list of (x, y) coordinates of the entire problem
        non_outer_wall_coords: list of (x, y) coordinates of the entire problem,
            excluding the outer border (these are the actual squares pacman can
            possibly be in)
        walls_grid: 2D array of either -1/0/1 or T/F. Used only for successorAxioms.
            Do NOT use this when making possible locations for pacman to be in.
        sensorModel(t, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
        successorAxioms(t, walls_grid, non_outer_wall_coords) -> Expr: function that generates
            the sensor model axioms. If None, it's not provided, so shouldn't be run.
    Return a logic sentence containing all of the following:
        - for all (x, y) in all_coords:
            If a wall is at (x, y) --> Pacman is not at (x, y)
        - Pacman is at exactly one of the squares at timestep t.
        - Pacman takes exactly one action at timestep t.
        - Results of calling sensorModel(...), unless None.
        - Results of calling successorAxioms(...), describing how Pacman can end in various
            locations on this time step. Consider edge cases. Don't call if None.
    """
    
    
    "*** BEGIN YOUR CODE HERE ***"
    pacphysics_sentences = [] 
    final_walls=[]  #A list to keep the expressions of walls
    final_pacman_position=[]  #A list to keep the expression of pacman's positions
   
    pacmanimplications=[] #A list that keeps for every (x,y) of all_coords the expression  (PropSymbolExpr(wall_str, x,y) >> ~ PropSymbolExpr(pacman_str, x,y,time=t)
    
    for x,y in all_coords:   #Creating the expressions for walls and pacman 
        final_walls.append(PropSymbolExpr(wall_str, x,y))
        final_pacman_position.append(PropSymbolExpr(pacman_str, x,y,time=t))
    
    squares_list=[] #Here we store the squares which the pacman can be at exactly one 
    for x,y in non_outer_wall_coords:            
        squares_list.append(PropSymbolExpr(pacman_str, x,y,time=t))
        
      
    for (iter4,iter5) in zip(final_walls,final_pacman_position): #Here we create the expression (PropSymbolExpr(wall_str, x,y) >> ~ PropSymbolExpr(pacman_str, x,y,time=t)
            pacmanimplications.append(iter4>> ~iter5) 
    
    direction=[]  #Here we store the action which pacman will do exactly one out of them
    for x in DIRECTIONS:
        direction.append(PropSymbolExpr(x, time=t))

    finalpacmanimplications=conjoin(pacmanimplications) #We conjoin ιτσ contents
    #squares=exactlyOne(squares_list)
    lit=[]   ###### From here the following process is exactly what exactlyone function does , basically we create the expression of 'pacman can be at only one square'and specifically we do that for the squares the pacman can be . 
    for x in squares_list :
        lit.append(~x)
    clauses=list(itertools.combinations(lit,2)) 
    lit2=[]
    lit3=[]
    final=[]    
    lit3=disjoin(squares_list)  
    final.append(lit3)                                  
    for x in clauses: 
        for y in x:
            lit2.append(y)
        final.append(atLeastOne(lit2))
        lit2.clear() 
    squares=conjoin(final) ###########################the process of having exactly one square that pacman can be in ends here 
    
    lit=[]               ##########Here starts the proccess of expressing , "pacman can take only one action".
    for x in direction :
        lit.append(~x)
    clauses=list(itertools.combinations(lit,2)) 
    lit2=[]
    lit3=[]
    final=[]    
    lit3=disjoin(direction)  
    final.append(lit3)                                  
    for x in clauses: 
        for y in x:
            lit2.append(y)
        final.append(atLeastOne(lit2))
        lit2.clear() 
    actions=conjoin(final) ### here ends the proccess of creating the expression pacman can takeo only one action 
    
    
    pacphysics_sentences.append(finalpacmanimplications) #I append the results in the list pacphysics_stentence .
    pacphysics_sentences.append(squares)
    pacphysics_sentences.append(actions)
    
       
    if sensorModel: #If sensormodel is not None then we append the return of sensormodel
        sensmod=sensorModel(t,non_outer_wall_coords)
        pacphysics_sentences.append(sensmod) 
                                                             #At this point i will explain what happens for the timestep t=0 in the readme file , and basically a further explanation will be added in the readme
    if successorAxioms and t!=0: #If successorAxioms is not None and the time is not zero then we append the return of successoraxioms
          pacphysics_sentences.append(successorAxioms(t, walls_grid, non_outer_wall_coords)) 
          
    
    
    return conjoin(pacphysics_sentences)
   # util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"
    


def checkLocationSatisfiability(x1_y1: Tuple[int, int], x0_y0: Tuple[int, int], action0, action1, problem):
    """
    Given:
        - x1_y1 = (x1, y1), a potential location at time t = 1
        - x0_y0 = (x0, y0), Pacman's location at time t = 0
        - action0 = one of the four items in DIRECTIONS, Pacman's action at time t = 0
        - action1 = to ensure match with autograder solution
        - problem = an instance of logicAgents.LocMapProblem
    Note:
        - there's no sensorModel because we know everything about the world
        - the successorAxioms should be allLegalSuccessorAxioms where needed
    Return:
        - a model where Pacman is at (x1, y1) at time t = 1
        - a model where Pacman is not at (x1, y1) at time t = 1
    """
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))
    KB = []
    x0, y0 = x0_y0
    x1, y1 = x1_y1
    # We know which coords are walls:
    map_sent = [PropSymbolExpr(wall_str, x, y) for x, y in walls_list]
    KB.append(conjoin(map_sent))
    "*** BEGIN YOUR CODE HERE ***"
    ###################################gia timesteps ##################################  
    KB.append(pacphysicsAxioms(1,all_coords,non_outer_wall_coords,walls_grid,None,allLegalSuccessorAxioms)) # As it is give "Add to KB: pacphysics_axioms(...) with the appropriate timesteps. ".Here we do that for the timestep 0 and the timestep 1.
    KB.append(pacphysicsAxioms(0,all_coords,non_outer_wall_coords,walls_grid,None,allLegalSuccessorAxioms))
    ################################################################################### 
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0))  #Add to KB: Pacman’s current location (x0, y0)
    ################################################################################### 
    KB.append(PropSymbolExpr(action0, time=0)) #Add to KB: Pacman takes action0
    #####################################################################################  
    KB.append(PropSymbolExpr(action1, time=1)) #Add to KB: Pacman takes action1
    ################################################################################## 
    KB=conjoin(KB) #We conjoin whatever we have in KB
    ##################################################################################
    mod=PropSymbolExpr(pacman_str, x1, y1, time=1)  #In model1, Pacman is at (x1, y1) at time t = 1 given x0_y0, action0, action1, proving that it's possible that Pacman there. Notably, if model1 is False, we know Pacman is guaranteed to NOT be there.
    mod2=~PropSymbolExpr(pacman_str, x1, y1, time=1) #In model2, Pacman is NOT at (x1, y1) at time t = 1 given x0_y0, action0, action1, proving that it's possible that Pacman is not there. Notably, if model2 is False, we know Pacman is guaranteed to be there.
    ###################################################################################
    final1=conjoin([KB,mod])  #We conjoin KB with mod at final1 for the first case 
    final2=conjoin([KB,mod2]) #We conjoin KB with mod2 at final2 for the second case 
    result=[]
    result.append(findModel(final1)) #Query the SAT solver with findModel for two models described earlier.We do that as we do it in entails
    result.append(findModel(final2))
    
    return tuple(result) #We return the result in a tuple.
    
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 4

def positionLogicPlan(problem) -> List:
    """
    Given an instance of a PositionPlanningProblem, return a list of actions that lead to the goal.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls_grid = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls_grid.asList()
    x0, y0 = problem.startState
    xg, yg = problem.goal
    
    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), 
            range(height + 2)))
    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]
    KB = []
    "*** BEGIN YOUR CODE HERE ***"
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0)) #Add to KB: Initial knowledge: Pacman's initial location at timestep 0
    for t in range(50):  #for t in range(50) [Autograder will not test on layouts requiring ≥50 timesteps]
    ######################################################################################################   #This process differs for t=0 and for t>0 , i will explain this further in readme why this happens.
        print("Timestep",t)   #Print time step; this is to see that the code is running and how far it is
        if t==0:   #For t=0 the following happen
            location_list=[]
            a_list=[]
            for iter2 in non_wall_coords:  #We make every location that pacman can be , into an expression 
                for iter3 in iter2:
                    a_list.append(iter3)
                location_list.append(PropSymbolExpr(pacman_str, a_list[0],a_list[1],time=t)) 
                a_list.clear()    #The end of making everh possible position pacman can be to an expression
                ################################################# here stops the process of creating the expressions of where pacman can be at 
            lit=[]
            for x in location_list :#######################here starts the proccess of " Pacman can only be at exactlyOne of the locations in non_wall_coords at timestep t." ,Basically i do exactly what i do in the function exactly one in order to express  Pacman can only be at exactlyOne of the locations in non_wall_coords at timestep t.
                lit.append(~x)
            clauses=list(itertools.combinations(lit,2)) 
            lit2=[]
            lit3=[]                                                  ########locations for pacman######
            final=[]    
            lit3=disjoin(location_list)  
            final.append(lit3)                                  
            for x in clauses: 
                for y in x:
                    lit2.append(y)
                final.append(atLeastOne(lit2))
                lit2.clear()                   ##here ends the proccess of expressing the pacman can only be at exactly one of the locations in non_wall_coords at timestep t
            KB.append(conjoin(final))    #I conjoin the result and insert it to the knowledge base
         ##############################################################################################################   
            KB=conjoin(KB)           #Here basically we check if with what we have in the knowledge base we can say that we indeed are in the goal state at t=0 if we are in goal state then we just return None because there were no actions taken , else we continue.
            goal_state=conjoin([KB,PropSymbolExpr(pacman_str, xg, yg, time=t)])     
            result=findModel(goal_state)
            if result is not False:
                 return None 
         #####################################################################################################################  
            action_list=[]  #Here i do the process were : I Add to KB: Pacman takes exactly one action per timestep.
            for iter4 in actions:  #We make every location that pacman can be into an expression 
                action_list.append(PropSymbolExpr(iter4,time=t))
            
            lit4=[]
            for x in action_list :  #Here starts the process of expressing pacman can only take only one action
                lit4.append(~x)
            clauses2=list(itertools.combinations(lit4,2)) 
            lit5=[]
            #lit6=[]                                                    
            final2=[]    
            lit6=disjoin(action_list)  
            final2.append(lit6)                                  
            for x in clauses2: 
                for y in x:
                    lit5.append(y)
                final2.append(atLeastOne(lit5))
                lit5.clear()
            final2=conjoin(final2)
            KB=[KB,final2]   #We add the result to KB
         ###############################################################################################################   
        if t>0:    #For t > 0 we do exactly the same as we did for t=0
            location_list=[]
            a_list=[]
            for iter2 in non_wall_coords:  #We make every location that pacman can be into an expression 
                for iter3 in iter2:
                    a_list.append(iter3)
                location_list.append(PropSymbolExpr(pacman_str, a_list[0],a_list[1],time=t)) 
                a_list.clear()
            lit=[]
            for x in location_list :  #We are expressing that pacman can be at exactly one position
                lit.append(~x)
            clauses=list(itertools.combinations(lit,2)) 
            lit2=[]
            lit3=[]                                                      ########locations for pacman######
            final=[]    
            lit3=disjoin(location_list)  
            final.append(lit3)                                  
            for x in clauses: 
                for y in x:
                    lit2.append(y)
                final.append(atLeastOne(lit2))
                lit2.clear()
            KB.append(conjoin(final))
         ########################################################################################################
            a_list=[]   #Add to KB: Transition Model sentences: call  
            for iter2 in non_wall_coords:            #########pacman successor################
                for iter3 in iter2:
                    a_list.append(iter3)  #Basically pacmansuccessorAxiomSingle is used in order to know how pacman got there at time t , waht action did it took?
                if t>0:                          #sigoyra meta to mhden sigoyra 
                    KB.append(pacmanSuccessorAxiomSingle(a_list[0],a_list[1],t,walls_grid))
                    
                a_list.clear()
         #########################################################################################################
            KB=conjoin(KB)    #Here with every information we have in KB we check if there can be a sequence of actions that can happen in order to get to the goal , if there is find it with find with extractActionSequence and return it , in order to do that we have to know if there is a satisfying model first.
            goal_state=conjoin([KB,PropSymbolExpr(pacman_str, xg, yg, time=t)])     #goal???
            result=findModel(goal_state)
            if result is not False:
                return extractActionSequence(result,actions) 
         ############################################################################################################
            action_list=[]  #Here we add to the knowledge base the fact that pacman can take exactly one action in order to get to the next position.
            for iter4 in actions:  #We make every location that pacman can be into an expression 
                action_list.append(PropSymbolExpr(iter4,time=t))
            
            lit4=[]
            for x in action_list :
                lit4.append(~x)
            clauses2=list(itertools.combinations(lit4,2)) 
            lit5=[]
                                                                
            final2=[]    
            lit6=disjoin(action_list)  
            final2.append(lit6)                                  
            for x in clauses2: 
                for y in x:
                    lit5.append(y)
                final2.append(atLeastOne(lit5))
                lit5.clear()
            final2=conjoin(final2)
            KB=[KB,final2]
         ##########################################################################################################         
    #util.raiseNotDefined()
    
    "*** END YOUR CODE HERE ***"
   
#______________________________________________________________________________
# QUESTION 5

def foodLogicPlan(problem) -> List:
    """
    Given an instance of a FoodPlanningProblem, return a list of actions that help Pacman
    eat all of the food.
    Available actions are ['North', 'East', 'South', 'West']
    Note that STOP is not an available action.
    Overview: add knowledge incrementally, and query for a model each timestep. Do NOT use pacphysicsAxioms.
    """
    walls = problem.walls
    width, height = problem.getWidth(), problem.getHeight()
    walls_list = walls.asList()
    (x0, y0), food = problem.start
    food = food.asList()

    # Get lists of possible locations (i.e. without walls) and possible actions
    all_coords = list(itertools.product(range(width + 2), range(height + 2)))

    non_wall_coords = [loc for loc in all_coords if loc not in walls_list]
    actions = [ 'North', 'South', 'East', 'West' ]

    KB = []

    "*** BEGIN YOUR CODE HERE ***"  #Here we basically do the same as in question4 but with some changes.
    KB.append(PropSymbolExpr(pacman_str, x0, y0, time=0)) #x0 ,y0 2,2 na ksereis mporei na thelei allagh
    #Initialize Food[x,y]_t variables with the code PropSymbolExpr(food_str, x, y, time=t), where each variable is true if and only if there is a food at (x, y) at time t.
    for x, y in non_wall_coords:
            KB.append(PropSymbolExpr(food_str, x, y, time=0) if (x, y) in food else ~PropSymbolExpr(food_str, x, y, time=0)) # thelei  allagh
    for t in range(50):        
    ######################################################################################################    
        if t==0:
         ############################################################################################################################################
           
         ##############################################################################################################################################
            location_list=[]    
            a_list=[]
            for iter2 in non_wall_coords:  #We make every location that pacman can be into an expression 
                for iter3 in iter2:
                    a_list.append(iter3)
                location_list.append(PropSymbolExpr(pacman_str, a_list[0],a_list[1],time=t)) 
                a_list.clear()
            lit=[]
            for x in location_list :  #Pacman can be at exactly one position 
                lit.append(~x)
            clauses=list(itertools.combinations(lit,2)) 
            lit2=[]
            lit3=[]                                                     ########locations for pacman######
            final=[]                                                                            
            lit3=disjoin(location_list)  
            final.append(lit3)                                  
            for x in clauses: 
                for y in x:
                    lit2.append(y)
                final.append(atLeastOne(lit2))
                lit2.clear()
            KB.append(conjoin(final))  # we insert it to the knowledge base
         ##############################################################################################################   
            KB=conjoin(KB)   
            
            goal_list=[]
            for iter6 in food:    #The goal now is different from quesgtion4.Change the goal assertion: Your goal assertion sentence must be true if and only if all of the food have been eaten. This happens when all Food[x,y]_t are false.
                    for iter7 in iter6:
                        a_list.append(iter7)
                    goal_list.append(~PropSymbolExpr(food_str, a_list[0],a_list[1],time=t)) 
                    a_list.clear()
            goal_list=conjoin(goal_list)  #We insert the goal to the knowledge base 
           
            goal_list=conjoin([goal_list,KB])   #If there is a satisfying model then we return None because no actions were made at time t=0
            result=findModel(goal_list)     
            
            if result is not False:
                 return None 
         #####################################################################################################################  
            action_list=[]
            for iter4 in actions:  #We make every action in to expression
                action_list.append(PropSymbolExpr(iter4,time=t))
            
            lit4=[]      #Pacman can take exactly one action , thats what we are doing here 
            for x in action_list :
                lit4.append(~x)
            clauses2=list(itertools.combinations(lit4,2))                   
            lit5=[]
                                                                
            final2=[]    
            lit6=disjoin(action_list)  
            final2.append(lit6)                                  
            for x in clauses2: 
                for y in x:
                    lit5.append(y)
                final2.append(atLeastOne(lit5))
                lit5.clear()
            final2=conjoin(final2)
            KB=[KB,final2]
         ###############################################################################################################
            a_list=[]   #Here is the transition model , basically the relation between Food[x,y]_t+1 and Food[x,y]_t and Pacman[x,y]_t is : (~Pacman[x,y]_t & Food[x,y]_t) -> Food[x,y]_t+1. So we express that and we insert it in the knowledge base in order to get the transition 
            
            for x, y in non_wall_coords:
                food_var = PropSymbolExpr(food_str, x, y, time=t)
                next_food_var = PropSymbolExpr(food_str, x, y, time=t+1) #if (x,y) not in food else food_var
                successor_expr = (~PropSymbolExpr(pacman_str, x, y, time=t)&food_var)>>next_food_var
                
                KB.append(successor_expr)
            
            
         ############################################################################################################################
        if t>0:  #We do the same for t>0 also 
         ##############################################################################################################################################
            location_list=[]
            a_list=[]
            for iter2 in non_wall_coords:  #We make every location that pacman can be into an expression 
                for iter3 in iter2:
                    a_list.append(iter3)
                location_list.append(PropSymbolExpr(pacman_str, a_list[0],a_list[1],time=t)) 
                a_list.clear()
            lit=[]
            for x in location_list :
                lit.append(~x)    #Pacman can only be at exactly one location
            clauses=list(itertools.combinations(lit,2)) 
            lit2=[]
            lit3=[]                                  ########locations for pacman######
            final=[]                                                                            
            lit3=disjoin(location_list)  
            final.append(lit3)                                  
            for x in clauses: 
                for y in x:
                    lit2.append(y)
                final.append(atLeastOne(lit2))
                lit2.clear()
            KB.append(conjoin(final))
            ########################################################################################################
            a_list=[]          #We add the transition for pacman in order to get there at time t 
            for iter2 in non_wall_coords:            #########pacman successor################
                for iter3 in iter2:
                    a_list.append(iter3)
                if t>0:                          #sigoyra meta to mhden sigoyra 
                    KB.append(pacmanSuccessorAxiomSingle(a_list[0],a_list[1],t,walls))
                    
                a_list.clear()
         ##############################################################################################################   
            KB=conjoin(KB)   
            
            goal_list=[]     #If there is a satisfying model then we can tell that there is a sequence of actions that can happen ion order to get to the goal
            for iter6 in food:    
                    for iter7 in iter6:
                        a_list.append(iter7)
                    goal_list.append(~PropSymbolExpr(food_str, a_list[0],a_list[1],time=t)) 
                    a_list.clear()
            goal_list=conjoin(goal_list)
          
            goal_list=conjoin([goal_list,KB])
            result=findModel(goal_list)
            
            if result is not False:
                 return extractActionSequence(result,actions) 
         #####################################################################################################################  
            action_list=[]
            for iter4 in actions:  #We make every action that pacman can take into an expression 
                action_list.append(PropSymbolExpr(iter4,time=t))
            
            lit4=[]
            for x in action_list :   #Here we do the proccess of pacman can take exactly one action
                lit4.append(~x)
            clauses2=list(itertools.combinations(lit4,2))                   
            lit5=[]
            #lit6=[]                                                    
            final2=[]    
            lit6=disjoin(action_list)  
            final2.append(lit6)                                  
            for x in clauses2: 
                for y in x:
                    lit5.append(y)
                final2.append(atLeastOne(lit5))
                lit5.clear()
            final2=conjoin(final2)
            KB=[KB,final2]
         ###############################################################################################################
            a_list=[]   #Here as in time t=0 we do the transition for the positions that have food 

            a_list=[] 
            for x, y in non_wall_coords:
                food_var = PropSymbolExpr(food_str, x, y, time=t)
                next_food_var = PropSymbolExpr(food_str, x, y, time=t+1) #if (x,y) not in food else food_var
                successor_expr = (~PropSymbolExpr(pacman_str, x, y, time=t)&food_var)>>next_food_var               
                KB.append(successor_expr)                
         ############################################################################################################################
         
         
    #util.raiseNotDefined()
    "*** END YOUR CODE HERE ***"

#______________________________________________________________________________
# QUESTION 6

def localization(problem, agent) -> Generator:
    '''
    problem: a LocalizationProblem instance
    agent: a LocalizationLogicAgent instance
    '''
    walls_grid = problem.walls
    walls_list = walls_grid.asList()
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    KB = []

    "*** BEGIN YOUR CODE HERE ***"
    only_walls=[] #This list holds only positions with walls 
    not_walls=[] #This list holds position that there is no wall 
    ############################################# #Add to KB: 
    for x, y in all_coords:      
        if (x,y) in walls_list:   #where the walls are (walls_list)
            only_walls.append(PropSymbolExpr(wall_str, x, y))
        if (x,y) not in walls_list:  #and aren't (not in walls_list)
            not_walls.append(~PropSymbolExpr(wall_str, x, y))
    
    KB.append(conjoin(only_walls))  #Add to KB: where the walls are (walls_list) and aren't (not in walls_list)
    KB.append(conjoin(not_walls))      
    ##################################################
    #util.raiseNotDefined()
    for t in range(agent.num_timesteps):
        #Here we Add pacphysics, action, and percept information to KB. pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
        KB.append(pacphysicsAxioms(t,all_coords,non_outer_wall_coords,walls_grid,sensorAxioms,allLegalSuccessorAxioms)) #Add to KB: pacphysics_axioms(...), which you wrote in q3. Use sensorAxioms and allLegalSuccessorAxioms for localization
        KB.append(PropSymbolExpr(agent.actions[t], time=t))#Add to KB: Pacman takes action prescribed by agent.actions[t]
        KB.append(fourBitPerceptRules(t,agent.getPercepts()))#Get the percepts by calling agent.getPercepts() and pass the percepts to fourBitPerceptRules(...) for localization
        #Here ends the helper function Add pacphysics, action, and percept information to KB and the function Find possible pacman locations with updated KB starts
        possible_locations = []
        for x,y in non_outer_wall_coords:
            result1=entails(conjoin(KB), PropSymbolExpr(pacman_str, x, y, time=t)) #Can we prove pacman is at x,y
            result2=entails(conjoin(KB), ~PropSymbolExpr(pacman_str, x, y, time=t)) #Can we prove pacman is not at x,y
            result3=findModel(conjoin(conjoin(KB),PropSymbolExpr(pacman_str, x, y, time=t)))
            if  result3!=False  : #prosoxh If there exists a satisfying assignment where Pacman is at (x, y) at time t, add (x, y) to possible_locations.
                possible_locations.append((x,y))
            if result1==True:
                KB.append(PropSymbolExpr(pacman_str, x, y, time=t))
            elif result2==True:
                KB.append(~PropSymbolExpr(pacman_str, x, y, time=t))
         #Here ends the function Find possible pacman locations with updated KB   
        agent.moveToNextState(agent.actions[t])   #Call agent.moveToNextState(action_t) on the current agent action at timestep t.
        #"*** END YOUR CODE HERE ***"
        
        yield possible_locations #yield the possible locations.
#______________________________________________________________________________
# QUESTION 7

def mapping(problem, agent) -> Generator:
    '''
    problem: a MappingProblem instance
    agent: a MappingLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]

    # Pacman knows that the outer border of squares are all walls
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***" #Get initial location (pac_x_0, pac_y_0) of Pacman, and add this to KB. Also add whether there is a wall at that location.
    KB.append(PropSymbolExpr(pacman_str,pac_x_0,pac_y_0, time=0))  #If there a wall then we have to update the known map with 1
    if known_map[pac_x_0][pac_y_0]==1:
        KB.append(PropSymbolExpr(wall_str, pac_x_0, pac_y_0))
    #util.raiseNotDefined()
    if known_map[pac_x_0][pac_y_0]==0:  #If there is not a wall then we have to update the known map with 0
        KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))
    
    for t in range(agent.num_timesteps):  #for t in range(agent.num_timesteps):
        #Here we Add pacphysics, action, and percept information to KB. pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
        KB.append(pacphysicsAxioms(t,all_coords,non_outer_wall_coords,known_map,sensorAxioms,allLegalSuccessorAxioms)) #Add to KB: pacphysics_axioms(...), which you wrote in q3. Use sensorAxioms and allLegalSuccessorAxioms for localization
        KB.append(PropSymbolExpr(agent.actions[t], time=t))#Add to KB: Pacman takes action prescribed by agent.actions[t]
        KB.append(fourBitPerceptRules(t,agent.getPercepts()))#Get the percepts by calling agent.getPercepts() and pass the percepts to fourBitPerceptRules(...) for localization
        #Here ends the helper function Add pacphysics, action, and percept information to KB and the function Find possible pacman locations with updated KB starts
        for x,y in non_outer_wall_coords:
            result1=entails(conjoin(KB), PropSymbolExpr(wall_str, x, y)) #Can we prove pacman is at x,y
            result2=entails(conjoin(KB), ~PropSymbolExpr(wall_str, x, y)) #Can we prove pacman is not at x,y
            if result1==True:
                KB.append(PropSymbolExpr(wall_str, x, y))
                known_map[x][y]=1
            if result2==True:
                KB.append(~PropSymbolExpr(pacman_str, x, y))
                known_map[x][y]=0
        #Here ends the function Find possible pacman locations with updated KB   
        agent.moveToNextState(agent.actions[t])  #Call agent.moveToNextState(action_t) on the current agent action at timestep t. 
        #"*** END YOUR CODE HERE ***"
        yield known_map #We yield known_map

#______________________________________________________________________________
# QUESTION 8

def slam(problem, agent) -> Generator:
    '''
    problem: a SLAMProblem instance
    agent: a SLAMLogicAgent instance
    '''
    pac_x_0, pac_y_0 = problem.startState
    KB = []
    all_coords = list(itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)))
    non_outer_wall_coords = list(itertools.product(range(1, problem.getWidth()+1), range(1, problem.getHeight()+1)))

    # map describes what we know, for GUI rendering purposes. -1 is unknown, 0 is open, 1 is wall
    known_map = [[-1 for y in range(problem.getHeight()+2)] for x in range(problem.getWidth()+2)]

    # We know that the outer_coords are all walls.
    outer_wall_sent = []
    for x, y in all_coords:
        if ((x == 0 or x == problem.getWidth() + 1)
                or (y == 0 or y == problem.getHeight() + 1)):
            known_map[x][y] = 1
            outer_wall_sent.append(PropSymbolExpr(wall_str, x, y))
    KB.append(conjoin(outer_wall_sent))

    "*** BEGIN YOUR CODE HERE ***"
    #util.raiseNotDefined()
    KB.append(PropSymbolExpr(pacman_str,pac_x_0,pac_y_0, time=0))  #Get initial location (pac_x_0, pac_y_0) of Pacman, and add this to KB. Update known_map accordingly and add the appropriate expression to KB.
    known_map[pac_x_0][pac_y_0]=0
    KB.append(~PropSymbolExpr(wall_str, pac_x_0, pac_y_0))

    for t in range(agent.num_timesteps):
        #Here we Add pacphysics, action, and percept information to KB. pacphysicsAxioms(t: int, all_coords: List[Tuple], non_outer_wall_coords: List[Tuple], walls_grid: List[List] = None, sensorModel: Callable = None, successorAxioms: Callable = None) -> Expr:
        KB.append(pacphysicsAxioms(t,all_coords,non_outer_wall_coords,known_map,SLAMSensorAxioms,SLAMSuccessorAxioms)) #Add to KB: pacphysics_axioms(...), which you wrote in q3. Use sensorAxioms and allLegalSuccessorAxioms for localization
        KB.append(PropSymbolExpr(agent.actions[t], time=t))#Add to KB: Pacman takes action prescribed by agent.actions[t]
        KB.append(numAdjWallsPerceptRules(t,agent.getPercepts()))#Get the percepts by calling agent.getPercepts() and pass the percepts to fourBitPerceptRules(...) for localization
        #Here ends the helper function Add pacphysics, action, and percept information to KB and the function Find possible pacman locations with updated KB starts
        possible_locations = []
        for x,y in non_outer_wall_coords:
            #Here starts the function Find provable wall locations with updated KB
            result1=entails(conjoin(KB), PropSymbolExpr(wall_str, x, y)) #Can we prove pacman is at x,y
            result2=entails(conjoin(KB), ~PropSymbolExpr(wall_str, x, y)) #Can we prove pacman is not at x,y
            if result1==True:
                KB.append(PropSymbolExpr(wall_str, x, y))
                known_map[x][y]=1
            if result2==True:
                KB.append(~PropSymbolExpr(pacman_str, x, y))
                known_map[x][y]=0
            #Here starts the function Find possible pacman locations with updated KB
            result1=entails(conjoin(KB), PropSymbolExpr(pacman_str, x, y, time=t)) #Can we prove pacman is at x,y
            result2=entails(conjoin(KB), ~PropSymbolExpr(pacman_str, x, y, time=t)) #Can we prove pacman is not at x,y
            result3=findModel(conjoin(conjoin(KB),PropSymbolExpr(pacman_str, x, y, time=t)))
            if  result3!=False  : #prosoxh If there exists a satisfying assignment where Pacman is at (x, y) at time t, add (x, y) to possible_locations.
                possible_locations.append((x,y))
            if result1==True:
                KB.append(PropSymbolExpr(pacman_str, x, y, time=t))
            elif result2==True:
                KB.append(~PropSymbolExpr(pacman_str, x, y, time=t))
        agent.moveToNextState(agent.actions[t])  #Call agent.moveToNextState(action_t) on the current agent action at timestep t.
        "*** END YOUR CODE HERE ***"
        yield (known_map, possible_locations)


# Abbreviations
plp = positionLogicPlan
loc = localization
mp = mapping
flp = foodLogicPlan
# Sometimes the logic module uses pretty deep recursion on long expressions
sys.setrecursionlimit(100000)

#______________________________________________________________________________
# Important expression generating functions, useful to read for understanding of this project.


def sensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (
                PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time = t)
        all_percept_exprs.append(percept_unit_clause % disjoin(percept_exprs))

    return conjoin(all_percept_exprs + combo_var_def_exprs)


def fourBitPerceptRules(t: int, percepts: List) -> Expr:
    """
    Localization and Mapping both use the 4 bit sensor, which tells us True/False whether
    a wall is to pacman's north, south, east, and west.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 4, "Percepts must be a length 4 list."

    percept_unit_clauses = []
    for wall_present, direction in zip(percepts, DIRECTIONS):
        percept_unit_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        if not wall_present:
            percept_unit_clause = ~PropSymbolExpr(blocked_str_map[direction], time=t)
        percept_unit_clauses.append(percept_unit_clause) # The actual sensor readings
    return conjoin(percept_unit_clauses)


def numAdjWallsPerceptRules(t: int, percepts: List) -> Expr:
    """
    SLAM uses a weaker numAdjWallsPerceptRules sensor, which tells us how many walls pacman is adjacent to
    in its four directions.
        000 = 0 adj walls.
        100 = 1 adj wall.
        110 = 2 adj walls.
        111 = 3 adj walls.
    """
    assert isinstance(percepts, list), "Percepts must be a list."
    assert len(percepts) == 3, "Percepts must be a length 3 list."

    percept_unit_clauses = []
    for i, percept in enumerate(percepts):
        n = i + 1
        percept_literal_n = PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t)
        if not percept:
            percept_literal_n = ~percept_literal_n
        percept_unit_clauses.append(percept_literal_n)
    return conjoin(percept_unit_clauses)


def SLAMSensorAxioms(t: int, non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    all_percept_exprs = []
    combo_var_def_exprs = []
    for direction in DIRECTIONS:
        percept_exprs = []
        dx, dy = DIR_TO_DXDY_MAP[direction]
        for x, y in non_outer_wall_coords:
            combo_var = PropSymbolExpr(pacman_wall_str, x, y, x + dx, y + dy, time=t)
            percept_exprs.append(combo_var)
            combo_var_def_exprs.append(combo_var % (PropSymbolExpr(pacman_str, x, y, time=t) & PropSymbolExpr(wall_str, x + dx, y + dy)))

        blocked_dir_clause = PropSymbolExpr(blocked_str_map[direction], time=t)
        all_percept_exprs.append(blocked_dir_clause % disjoin(percept_exprs))

    percept_to_blocked_sent = []
    for n in range(1, 4):
        wall_combos_size_n = itertools.combinations(blocked_str_map.values(), n)
        n_walls_blocked_sent = disjoin([
            conjoin([PropSymbolExpr(blocked_str, time=t) for blocked_str in wall_combo])
            for wall_combo in wall_combos_size_n])
        # n_walls_blocked_sent is of form: (N & S) | (N & E) | ...
        percept_to_blocked_sent.append(
            PropSymbolExpr(geq_num_adj_wall_str_map[n], time=t) % n_walls_blocked_sent)

    return conjoin(all_percept_exprs + combo_var_def_exprs + percept_to_blocked_sent)


def allLegalSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = pacmanSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)


def SLAMSuccessorAxioms(t: int, walls_grid: List[List], non_outer_wall_coords: List[Tuple[int, int]]) -> Expr:
    """walls_grid can be a 2D array of ints or bools."""
    all_xy_succ_axioms = []
    for x, y in non_outer_wall_coords:
        xy_succ_axiom = SLAMSuccessorAxiomSingle(
            x, y, t, walls_grid)
        if xy_succ_axiom:
            all_xy_succ_axioms.append(xy_succ_axiom)
    return conjoin(all_xy_succ_axioms)

#______________________________________________________________________________
# Various useful functions, are not needed for completing the project but may be useful for debugging


def modelToString(model: Dict[Expr, bool]) -> str:
    """Converts the model to a string for printing purposes. The keys of a model are 
    sorted before converting the model to a string.
    
    model: Either a boolean False or a dictionary of Expr symbols (keys) 
    and a corresponding assignment of True or False (values). This model is the output of 
    a call to pycoSAT.
    """
    if model == False:
        return "False" 
    else:
        # Dictionary
        modelList = sorted(model.items(), key=lambda item: str(item[0]))
        return str(modelList)


def extractActionSequence(model: Dict[Expr, bool], actions: List) -> List:
    """
    Convert a model in to an ordered list of actions.
    model: Propositional logic model stored as a dictionary with keys being
    the symbol strings and values being Boolean: True or False
    Example:
    >>> model = {"North[2]":True, "P[3,4,0]":True, "P[3,3,0]":False, "West[0]":True, "GhostScary":True, "West[2]":False, "South[1]":True, "East[0]":False}
    >>> actions = ['North', 'South', 'East', 'West']
    >>> plan = extractActionSequence(model, actions)
    >>> print(plan)
    ['West', 'South', 'North']
    """
    plan = [None for _ in range(len(model))]
    for sym, val in model.items():
        parsed = parseExpr(sym)
        if type(parsed) == tuple and parsed[0] in actions and val:
            action, _, time = parsed
            plan[time] = action
    #return list(filter(lambda x: x is not None, plan))
    return [x for x in plan if x is not None]


# Helpful Debug Method
def visualizeCoords(coords_list, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    for (x, y) in itertools.product(range(problem.getWidth()+2), range(problem.getHeight()+2)):
        if (x, y) in coords_list:
            wallGrid.data[x][y] = True
    print(wallGrid)


# Helpful Debug Method
def visualizeBoolArray(bool_arr, problem) -> None:
    wallGrid = game.Grid(problem.walls.width, problem.walls.height, initialValue=False)
    wallGrid.data = copy.deepcopy(bool_arr)
    print(wallGrid)

class PlanningProblem:
    """
    This class outlines the structure of a planning problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the planning problem.
        """
        util.raiseNotDefined()

    def getGhostStartStates(self):
        """
        Returns a list containing the start state for each ghost.
        Only used in problems that use ghosts (FoodGhostPlanningProblem)
        """
        util.raiseNotDefined()
        
    def getGoalState(self):
        """
        Returns goal state for problem. Note only defined for problems that have
        a unique goal state such as PositionPlanningProblem
        """
        util.raiseNotDefined()
