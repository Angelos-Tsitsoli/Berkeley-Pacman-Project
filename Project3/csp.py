"""CSP (Constraint Satisfaction Problems) problems and solvers. (Chapter 6)"""

import itertools
import random
import re
import string
from collections import defaultdict, Counter
from functools import reduce
from operator import eq, neg
from timeit import default_timer as timer
from sortedcontainers import SortedSet

import search
from utils import argmin_random_tie, count, first, extend


class CSP(search.Problem):
    """This class describes finite-domain Constraint Satisfaction Problems.
    A CSP is specified by the following inputs:
        variables   A list of variables; each is atomic (e.g. int or string).
        domains     A dict of {var:[possible_value, ...]} entries.
        neighbors   A dict of {var:[var,...]} that for each variable lists
                    the other variables that participate in constraints.
        constraints A function f(A, a, B, b) that returns true if neighbors
                    A, B satisfy the constraint when they have values A=a, B=b

    In the textbook and in most mathematical definitions, the
    constraints are specified as explicit pairs of allowable values,
    but the formulation here is easier to express and more compact for
    most cases (for example, the n-Queens problem can be represented
    in O(n) space using this notation, instead of O(n^4) for the
    explicit representation). In terms of describing the CSP as a
    problem, that's all there is.

    However, the class also supports data structures and methods that help you
    solve CSPs by calling a search function on the CSP. Methods and slots are
    as follows, where the argument 'a' represents an assignment, which is a
    dict of {var:val} entries:
        assign(var, val, a)     Assign a[var] = val; do other bookkeeping
        unassign(var, a)        Do del a[var], plus other bookkeeping
        nconflicts(var, val, a) Return the number of other variables that
                                conflict with var=val
        curr_domains[var]       Slot: remaining consistent values for var
                                Used by constraint propagation routines.
    The following methods are used only by graph_search and tree_search:
        actions(state)          Return a list of actions
        result(state, action)   Return a successor of state
        goal_test(state)        Return true if all constraints satisfied
    The following are just for debugging purposes:
        nassigns                Slot: tracks the number of assignments made
        display(a)              Print a human-readable representation
    """

    def __init__(self, variables, domains, neighbors, constraints):
        """Construct a CSP problem. If variables is empty, it becomes domains.keys()."""
        super().__init__(())
        variables = variables or list(domains.keys())
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.constraints = constraints
        self.curr_domains = None
        self.nassigns = 0

    def assign(self, var, val, assignment):
        """Add {var: val} to assignment; Discard the old value if any."""
        assignment[var] = val
        self.nassigns += 1

    def unassign(self, var, assignment):
        """Remove {var: val} from assignment.
        DO NOT call this if you are changing a variable to a new value;
        just call assign for that."""
        if var in assignment:
            del assignment[var]

    def nconflicts(self, var, val, assignment):
        """Return the number of conflicts var=val has with other variables."""

        # Subclasses may implement this more efficiently
        def conflict(var2):
            return var2 in assignment and not self.constraints(var, val, var2, assignment[var2])

        return count(conflict(v) for v in self.neighbors[var])

    def display(self, assignment):
        """Show a human-readable representation of the CSP."""
        # Subclasses can print in a prettier way, or display with a GUI
        print(assignment)

    # These methods are for the tree and graph-search interface:

    def actions(self, state):
        """Return a list of applicable actions: non conflicting
        assignments to an unassigned variable."""
        if len(state) == len(self.variables):
            return []
        else:
            assignment = dict(state)
            var = first([v for v in self.variables if v not in assignment])
            return [(var, val) for val in self.domains[var]
                    if self.nconflicts(var, val, assignment) == 0]

    def result(self, state, action):
        """Perform an action and return the new state."""
        (var, val) = action
        return state + ((var, val),)

    def goal_test(self, state):
        """The goal is to assign all variables, with all constraints satisfied."""
        assignment = dict(state)
        return (len(assignment) == len(self.variables)
                and all(self.nconflicts(variables, assignment[variables], assignment) == 0
                        for variables in self.variables))

    # These are for constraint propagation

    def support_pruning(self):
        """Make sure we can prune values from domains. (We want to pay
        for this only if we use it.)"""
        if self.curr_domains is None:
            self.curr_domains = {v: list(self.domains[v]) for v in self.variables}

    def suppose(self, var, value):
        """Start accumulating inferences from assuming var=value."""
        self.support_pruning()
        removals = [(var, a) for a in self.curr_domains[var] if a != value]
        self.curr_domains[var] = [value]
        return removals

    def prune(self, var, value, removals):
        """Rule out var=value."""
        self.curr_domains[var].remove(value)
        if removals is not None:
            removals.append((var, value))

    def choices(self, var):
        """Return all values for var that aren't currently ruled out."""
        return (self.curr_domains or self.domains)[var]

    def infer_assignment(self):
        """Return the partial assignment implied by the current inferences."""
        self.support_pruning()
        return {v: self.curr_domains[v][0]
                for v in self.variables if 1 == len(self.curr_domains[v])}

    def restore(self, removals):
        """Undo a supposition and all inferences from it."""
        for B, b in removals:
            self.curr_domains[B].append(b)

    # This is for min_conflicts search

    def conflicted_vars(self, current):
        """Return a list of variables in current assignment that are in conflict"""
        return [var for var in self.variables
                if self.nconflicts(var, current[var], current) > 0]


# ______________________________________________________________________________
# Constraint Propagation with AC3


def no_arc_heuristic(csp, queue):
    return queue


def dom_j_up(csp, queue):
    return SortedSet(queue, key=lambda t: neg(len(csp.curr_domains[t[1]])))


def AC3(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    """[Figure 6.3]"""
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
   
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        revised, checks = revise(csp, Xi, Xj, removals, checks)
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
    return True, checks  # CSP is satisfiable


def revise(csp, Xi, Xj, removals, checks=0):
    """Return true if we remove a value."""
    revised = False
    for x in csp.curr_domains[Xi][:]:
        # If Xi=x conflicts with Xj=y for every possible y, eliminate Xi=x
        # if all(not csp.constraints(Xi, x, Xj, y) for y in csp.curr_domains[Xj]):
        conflict = True
        for y in csp.curr_domains[Xj]:
            if csp.constraints(Xi, x, Xj, y):
                conflict = False
            checks += 1
            if not conflict:
                break
        if conflict:
            csp.prune(Xi, x, removals)
            revised = True
    return revised, checks


# Constraint Propagation with AC3b: an improved version
# of AC3 with double-support domain-heuristic

def AC3b(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    checks = 0
    while queue:
        (Xi, Xj) = queue.pop()
        # Si_p values are all known to be supported by Xj
        # Sj_p values are all known to be supported by Xi
        # Dj - Sj_p = Sj_u values are unknown, as yet, to be supported by Xi
        Si_p, Sj_p, Sj_u, checks = partition(csp, Xi, Xj, checks)
        if not Si_p:
            return False, checks  # CSP is inconsistent
        revised = False
        for x in set(csp.curr_domains[Xi]) - Si_p:
            csp.prune(Xi, x, removals)
            revised = True
        if revised:
            for Xk in csp.neighbors[Xi]:
                if Xk != Xj:
                    queue.add((Xk, Xi))
        if (Xj, Xi) in queue:
            if isinstance(queue, set):
                # or queue -= {(Xj, Xi)} or queue.remove((Xj, Xi))
                queue.difference_update({(Xj, Xi)})
            else:
                queue.difference_update((Xj, Xi))
            # the elements in D_j which are supported by Xi are given by the union of Sj_p with the set of those
            # elements of Sj_u which further processing will show to be supported by some vi_p in Si_p
            for vj_p in Sj_u:
                for vi_p in Si_p:
                    conflict = True
                    if csp.constraints(Xj, vj_p, Xi, vi_p):
                        conflict = False
                        Sj_p.add(vj_p)
                    checks += 1
                    if not conflict:
                        break
            revised = False
            for x in set(csp.curr_domains[Xj]) - Sj_p:
                csp.prune(Xj, x, removals)
                revised = True
            if revised:
                for Xk in csp.neighbors[Xj]:
                    if Xk != Xi:
                        queue.add((Xk, Xj))
    return True, checks  # CSP is satisfiable


def partition(csp, Xi, Xj, checks=0):
    Si_p = set()
    Sj_p = set()
    Sj_u = set(csp.curr_domains[Xj])
    for vi_u in csp.curr_domains[Xi]:
        conflict = True
        # now, in order to establish support for a value vi_u in Di it seems better to try to find a support among
        # the values in Sj_u first, because for each vj_u in Sj_u the check (vi_u, vj_u) is a double-support check
        # and it is just as likely that any vj_u in Sj_u supports vi_u than it is that any vj_p in Sj_p does...
        for vj_u in Sj_u - Sj_p:
            # double-support check
            if csp.constraints(Xi, vi_u, Xj, vj_u):
                conflict = False
                Si_p.add(vi_u)
                Sj_p.add(vj_u)
            checks += 1
            if not conflict:
                break
        # ... and only if no support can be found among the elements in Sj_u, should the elements vj_p in Sj_p be used
        # for single-support checks (vi_u, vj_p)
        if conflict:
            for vj_p in Sj_p:
                # single-support check
                if csp.constraints(Xi, vi_u, Xj, vj_p):
                    conflict = False
                    Si_p.add(vi_u)
                checks += 1
                if not conflict:
                    break
    return Si_p, Sj_p, Sj_u - Sj_p, checks


# Constraint Propagation with AC4

def AC4(csp, queue=None, removals=None, arc_heuristic=dom_j_up):
    if queue is None:
        queue = {(Xi, Xk) for Xi in csp.variables for Xk in csp.neighbors[Xi]}
    csp.support_pruning()
    queue = arc_heuristic(csp, queue)
    support_counter = Counter()
    variable_value_pairs_supported = defaultdict(set)
    unsupported_variable_value_pairs = []
    checks = 0
    # construction and initialization of support sets
    while queue:
        (Xi, Xj) = queue.pop()
        revised = False
        for x in csp.curr_domains[Xi][:]:
            for y in csp.curr_domains[Xj]:
                if csp.constraints(Xi, x, Xj, y):
                    support_counter[(Xi, x, Xj)] += 1
                    variable_value_pairs_supported[(Xj, y)].add((Xi, x))
                checks += 1
            if support_counter[(Xi, x, Xj)] == 0:
                csp.prune(Xi, x, removals)
                revised = True
                unsupported_variable_value_pairs.append((Xi, x))
        if revised:
            if not csp.curr_domains[Xi]:
                return False, checks  # CSP is inconsistent
    # propagation of removed values
    while unsupported_variable_value_pairs:
        Xj, y = unsupported_variable_value_pairs.pop()
        for Xi, x in variable_value_pairs_supported[(Xj, y)]:
            revised = False
            if x in csp.curr_domains[Xi][:]:
                support_counter[(Xi, x, Xj)] -= 1
                if support_counter[(Xi, x, Xj)] == 0:
                    csp.prune(Xi, x, removals)
                    revised = True
                    unsupported_variable_value_pairs.append((Xi, x))
            if revised:
                if not csp.curr_domains[Xi]:
                    return False, checks  # CSP is inconsistent
    return True, checks  # CSP is satisfiable


# ______________________________________________________________________________
# CSP Backtracking Search

# Variable ordering


def first_unassigned_variable(assignment, csp):
    """The default variable order."""
    return first([var for var in csp.variables if var not in assignment])


def mrv(assignment, csp):
    """Minimum-remaining-values heuristic."""
    return argmin_random_tie([v for v in csp.variables if v not in assignment],
                             key=lambda var: num_legal_values(csp, var, assignment))


def num_legal_values(csp, var, assignment):
    if csp.curr_domains:
        return len(csp.curr_domains[var])
    else:
        return count(csp.nconflicts(var, val, assignment) == 0 for val in csp.domains[var])


# Value ordering


def unordered_domain_values(var, assignment, csp):
    """The default value order."""
    return csp.choices(var)


def lcv(var, assignment, csp):
    """Least-constraining-values heuristic."""
    return sorted(csp.choices(var), key=lambda val: csp.nconflicts(var, val, assignment))


# Inference


def no_inference(csp, var, value, assignment, removals):
    return True


def forward_checking(csp, var, value, assignment, removals):
    """Prune neighbor values inconsistent with var=value."""
    csp.support_pruning()
    for B in csp.neighbors[var]:
        if B not in assignment:
            for b in csp.curr_domains[B][:]:
                if not csp.constraints(var, value, B, b):
                    csp.prune(B, b, removals)
            if not csp.curr_domains[B]:
                return False
    return True


def mac(csp, var, value, assignment, removals, constraint_propagation=AC3b):
    """Maintain arc consistency."""
    return constraint_propagation(csp, {(X, var) for X in csp.neighbors[var]}, removals)


# The search, proper


def backtracking_search(csp, select_unassigned_variable=first_unassigned_variable,
                        order_domain_values=unordered_domain_values, inference=no_inference):
    """[Figure 6.5]"""
    
    def backtrack(assignment):
        if len(assignment) == len(csp.variables):
            return assignment
        var = select_unassigned_variable(assignment, csp)
        #print(assignment)
        #print(var)
        for value in order_domain_values(var, assignment, csp):
           # print(value)
            if 0 == csp.nconflicts(var, value, assignment):
                csp.assign(var, value, assignment)
                removals = csp.suppose(var, value)
                #print(removals)
                if inference(csp, var, value, assignment, removals):
                    result = backtrack(assignment)
                    if result is not None:
                        return result
                csp.restore(removals)
        csp.unassign(var, assignment)
        
        return None

    result = backtrack({})
    assert result is None or csp.goal_test(result)
  
    return result


# ______________________________________________________________________________
# Min-conflicts Hill Climbing search for CSPs


def min_conflicts(csp, max_steps=100000):
    """Solve a CSP by stochastic Hill Climbing on the number of conflicts."""
    # Generate a complete assignment for all variables (probably with conflicts)
    csp.current = current = {}
    for var in csp.variables:
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    # Now repeatedly choose a random conflicted variable and change it
    for i in range(max_steps):
        conflicted = csp.conflicted_vars(current)
        if not conflicted:
            return current
        var = random.choice(conflicted)
        val = min_conflicts_value(csp, var, current)
        csp.assign(var, val, current)
    return None


def min_conflicts_value(csp, var, current):
    """Return the value that will give var the least number of conflicts.
    If there is a tie, choose at random."""
    return argmin_random_tie(csp.domains[var], key=lambda val: csp.nconflicts(var, val, current))


# ______________________________________________________________________________


def tree_csp_solver(csp):
    """[Figure 6.11]"""
    assignment = {}
    root = csp.variables[0]
    X, parent = topological_sort(csp, root)

    csp.support_pruning()
    for Xj in reversed(X[1:]):
        if not make_arc_consistent(parent[Xj], Xj, csp):
            return None

    assignment[root] = csp.curr_domains[root][0]
    for Xi in X[1:]:
        assignment[Xi] = assign_value(parent[Xi], Xi, csp, assignment)
        if not assignment[Xi]:
            return None
    return assignment


def topological_sort(X, root):
    """Returns the topological sort of X starting from the root.

    Input:
    X is a list with the nodes of the graph
    N is the dictionary with the neighbors of each node
    root denotes the root of the graph.

    Output:
    stack is a list with the nodes topologically sorted
    parents is a dictionary pointing to each node's parent

    Other:
    visited shows the state (visited - not visited) of nodes

    """
    neighbors = X.neighbors

    visited = defaultdict(lambda: False)

    stack = []
    parents = {}

    build_topological(root, None, neighbors, visited, stack, parents)
    return stack, parents


def build_topological(node, parent, neighbors, visited, stack, parents):
    """Build the topological sort and the parents of each node in the graph."""
    visited[node] = True

    for n in neighbors[node]:
        if not visited[n]:
            build_topological(n, node, neighbors, visited, stack, parents)

    parents[node] = parent
    stack.insert(0, node)


def make_arc_consistent(Xj, Xk, csp):
    """Make arc between parent (Xj) and child (Xk) consistent under the csp's constraints,
    by removing the possible values of Xj that cause inconsistencies."""
    # csp.curr_domains[Xj] = []
    for val1 in csp.domains[Xj]:
        keep = False  # Keep or remove val1
        for val2 in csp.domains[Xk]:
            if csp.constraints(Xj, val1, Xk, val2):
                # Found a consistent assignment for val1, keep it
                keep = True
                break

        if not keep:
            # Remove val1
            csp.prune(Xj, val1, None)

    return csp.curr_domains[Xj]


def assign_value(Xj, Xk, csp, assignment):
    """Assign a value to Xk given Xj's (Xk's parent) assignment.
    Return the first value that satisfies the constraints."""
    parent_assignment = assignment[Xj]
    for val in csp.curr_domains[Xk]:
        if csp.constraints(Xj, parent_assignment, Xk, val):
            return val

    # No consistent assignment available
    return None


# ______________________________________________________________________________
# Map Coloring CSP Problems


class UniversalDict:
    """A universal dict maps any key to the same value. We use it here
    as the domains dict for CSPs in which all variables have the same domain.
    >>> d = UniversalDict(42)
    >>> d['life']
    42
    """

    def __init__(self, value): self.value = value

    def __getitem__(self, key): return self.value

    def __repr__(self): return '{{Any: {0!r}}}'.format(self.value)


def different_values_constraint(A, a, B, b):
    """A constraint saying two neighboring variables must differ in value."""
    return a != b


def MapColoringCSP(colors, neighbors):
    """Make a CSP for the problem of coloring a map with different colors
    for any two adjacent regions. Arguments are a list of colors, and a
    dict of {region: [neighbor,...]} entries. This dict may also be
    specified as a string of the form defined by parse_neighbors."""
    if isinstance(neighbors, str):
        neighbors = parse_neighbors(neighbors)
    return CSP(list(neighbors.keys()), UniversalDict(colors), neighbors, different_values_constraint)


def parse_neighbors(neighbors):
    """Convert a string of the form 'X: Y Z; Y: Z' into a dict mapping
    regions to neighbors. The syntax is a region name followed by a ':'
    followed by zero or more region names, followed by ';', repeated for
    each region name. If you say 'X: Y' you don't need 'Y: X'.
    >>> parse_neighbors('X: Y Z; Y: Z') == {'Y': ['X', 'Z'], 'X': ['Y', 'Z'], 'Z': ['X', 'Y']}
    True
    """
    dic = defaultdict(list)
    specs = [spec.split(':') for spec in neighbors.split(';')]
    for (A, Aneighbors) in specs:
        A = A.strip()
        for B in Aneighbors.split():
            dic[A].append(B)
            dic[B].append(A)
    return dic


australia_csp = MapColoringCSP(list('RGB'), """SA: WA NT Q NSW V; NT: WA Q; NSW: Q V; T: """)

usa_csp = MapColoringCSP(list('RGBY'),
                         """WA: OR ID; OR: ID NV CA; CA: NV AZ; NV: ID UT AZ; ID: MT WY UT;
                         UT: WY CO AZ; MT: ND SD WY; WY: SD NE CO; CO: NE KA OK NM; NM: OK TX AZ;
                         ND: MN SD; SD: MN IA NE; NE: IA MO KA; KA: MO OK; OK: MO AR TX;
                         TX: AR LA; MN: WI IA; IA: WI IL MO; MO: IL KY TN AR; AR: MS TN LA;
                         LA: MS; WI: MI IL; IL: IN KY; IN: OH KY; MS: TN AL; AL: TN GA FL;
                         MI: OH IN; OH: PA WV KY; KY: WV VA TN; TN: VA NC GA; GA: NC SC FL;
                         PA: NY NJ DE MD WV; WV: MD VA; VA: MD DC NC; NC: SC; NY: VT MA CT NJ;
                         NJ: DE; DE: MD; MD: DC; VT: NH MA; MA: NH RI CT; CT: RI; ME: NH;
                         HI: ; AK: """)

france_csp = MapColoringCSP(list('RGBY'),
                            """AL: LO FC; AQ: MP LI PC; AU: LI CE BO RA LR MP; BO: CE IF CA FC RA
                            AU; BR: NB PL; CA: IF PI LO FC BO; CE: PL NB NH IF BO AU LI PC; FC: BO
                            CA LO AL RA; IF: NH PI CA BO CE; LI: PC CE AU MP AQ; LO: CA AL FC; LR:
                            MP AU RA PA; MP: AQ LI AU LR; NB: NH CE PL BR; NH: PI IF CE NB; NO:
                            PI; PA: LR RA; PC: PL CE LI AQ; PI: NH NO CA IF; PL: BR NB CE PC; RA:
                            AU BO FC PA LR""")


# ______________________________________________________________________________
# n-Queens Problem
                
                #This function is used by the Kenken class to read a line of the file kenken.txt specifically to take an example 
def reading(x): #The number that the variable x has tells what line of the kenken text we want .
    y=0          #Every line in the kenken test is a kenken example as we go down the file kenken we get more difficult tests by desceding order in the file we have examples:3x3,4x4,5x5,.....,9x9
    array=[]
    size=0
    f = open('kenken.txt','r')
    for line in f.readlines():  #When we find the line we want then we copy it to array and we break from the loop
     y=y+1
     if(y==x):
         array=line
         size=line[0]
         break
     
    return array,size
    

counter_const=0
class Kenken(CSP):
         
    def __init__(self,number): #In this function we initialize our csp problem by finding the variables,domains , neighbors .Init function has only one parameter which is the number of the example we want to use Kenken 
        self.number=number 
        clique_neighbor={}
        self.clique_neighbors=clique_neighbor   #The dictionary with the neighbors of every variable in clique
        size=0
        line=[]
        ch=[]
        line,size=reading(number) #line with store the line of the kenken.txt and size the size of the kenken puzzle 3x3,4x4...
        print(line) 
        
        clique=[]
        i2=0
        neighbors={}
        domains={}
        self.neighbors=neighbors #The dictionary of every variable with the neighbors it has in the row and in the column that it is in .
        self.domains=domains #The dictionary with the domains that can be chosen to fill the cells of the kenken puzzle
                               #The clique neghbors dictionary has every neighbor for every variable in the same clique 
                               #The goal list dictionary contains for every variable the calculation of the clique that this variable is in 
                               # the goal list2 does the same as the goal list dictionary but for every variable stores the the number which this clique has to reach
        a_list=[]
        a_list2=[]
        a_list5=[]
        a_list6=[]
        
        for iterator in range(1,10): 
           a_list5.append(str(iterator))
           
        for iterator in range(1,10):#Just a list with number ini order to check later if a variable has one of those numbers
           a_list6.append(iterator)
         
        for var in range(1,int(size)+1):   #we fill the dictionary variables , with the variables of the line in kenken.txt
            for var2 in range(1,int(size)+1):
                a_list2.append(var+(var2/10))         
        self.variables=a_list2
        
        for i in range(1,int(size)+1): #We see the front number of the line we want in kenken.txt to fill the dictionary domains .Basiccaly if that number is 3 then we knoe that the example is a 3x3 , if its 4 then a 4x4.If the number is 3x3 then we fill chwith the numbers 1,2,3
            ch.append(i)
        
        
        
        
        for var in range(1,(int(size)+1)):   #For every variable we give them as choices the list ch
             for var2 in range(1,int(size)+1): 
                 sum=var +(var2/10) 
                 self.domains[sum]=ch                        
        
        
        for i in range(1,int(size)+1):  #We find for every variable its neighbors in the same row and in column . This for loop checks the first part of a variable basically before the '.' basically it finds start with 1. the with 2. it is responsible for the rows
            for j in range(1,int(size)+1): #This for loop is respinsible for the columns
                i2=j/10        #We create the variable fr example 1.1, 2.2 ,1.3 
                i3=i2+i
                
                if i2==0.1:    #For the first column 0.1 we make the neighbors
                    
                    a_list=[]
                    for j2 in range(1,int(size)):  #It gives the 0.1 variable for example 1.1 , 2.1 , 3.1 .... its neighbors tat are after it like for 1.1 it gives 1.2 , 1.3 , 1.4 ......              
                        a_list.append(i+(j+j2)/10)
                    if(i==1):             #If its the first row which means its the variable 1.1 give as column neighbors every cell above it
                     for i4 in range(2,int(size)+1):
                       a_list.append(i4+i2)
                    if(i==int(size)) :   #If its the last  row give it as neighbors of  column every column abovw
                      
                      for i4 in range(int(size)-1,0,-1):  
                       a_list.append(i4+i2)
                    if(i!=int(size)and i!=1):  #If its not the first row or final row then give as neighbors the cells above it and below it 
                       
                       for i4 in range(i-1,0,-1):
                        a_list.append(i4+i2)
                       for i4 in range(i+1,int(size)+1):
                        a_list.append(i4+i2)
                              
                       
                    self.neighbors[i3]=a_list #Give the final neighbors 
                            
                if i2!=0.1 and i2!=int(size)/10:     #If the column we check is between the first and last column of the matrix the do the below to fill the neighbors         
                   a_list4=[]
                   
                   for j2 in range(j-1,0,-1):         #For the row neighbors go backwards and forward on the row to get the neighbors  
                        a_list4.append(i+(j2/10))      #Backwards
                        
                   for j2 in range(j+1,int(size)+1):   #   Forward          
                        a_list4.append(i+(j2)/10)
                        
                   if(i==1):                          #If the row is the first row
                     for i4 in range(2,int(size)+1):    #for the neighbors in the same column just go below from this cell to get the neighbors
                       a_list4.append(i4+i2)
                   if(i==int(size)) :                 #If the row is th last just go upwards from this cell to get the neighbors
                      for i4 in range(int(size)-1,0,-1):
                       a_list4.append(i4+i2)
                   if(i!=int(size)and i!=1):            #If the row is between the first and the last row then go upwards and downwards from this cell
                     
                       for i4 in range(i-1,0,-1):
                        a_list4.append(i4+i2)
                       for i4 in range(i+1,int(size)+1):
                        a_list4.append(i4+i2)
                        
                   self.neighbors[i3]=a_list4  #The final
                              
                if i2==int(size)/10:             #Iif the column is the last then do as follows
                   a_list3=[]
                   for j2 in range(1,int(size)):     #for the row neighbors just go backwards in the row          
                        a_list3.append(i+(j-j2)/10)
                   if(i==1):                           #for the column neighbors if we are at the first row then just go down in the matrix to get the neighbors
                     for i4 in range(2,int(size)+1):
                       a_list3.append(i4+i2)
                   if(i==int(size)) :                      #if we are at the last row then just go upwards to find the neighbors
                      for i4 in range(int(size)-1,0,-1):
                       a_list3.append(i4+i2)
                   if(i!=int(size)and i!=1):              #if we are between the first and the last row then go upwards and downwards
                     
                       for i4 in range(i-1,0,-1):
                        a_list3.append(i4+i2)
                       for i4 in range(i+1,int(size)+1):
                        a_list3.append(i4+i2)
                   self.neighbors[i3]=a_list3     #allaksa 
                 
                           #I did the following in order to get the neighbors that are in the same clique.
        charac1=' '
        charac2=' '
        charac3=' '
        number1=0
        number2=0 
        flag=0 
        backup_list=[]
        neighbors_list=[]
        iterator2=3
        charac1=line[iterator2]
        charac2=line[iterator2-1]
        charac3=line[iterator2-2]
        self.goal_list={}
        self.goal_list2={}
        subgoal_list2=[]
        subgoal_list=[]
        while charac1!='$' :        #while we havent read all the line of the kenken.txt file then do the following . The iteration of every character in the line we exirament happens with th variables charac1 which is the first that sees a character , with the charac2 which is the second iterator that follows chara1 behind and the charac3 which follows the other two.
             
               
           
             if charac1==']':        #If we see a closing bracket then this means we have seen a clique , so for every variable in that clique we have to give it its clique neighbors
                 
                for iterate in neighbors_list: # we iterate the neighbors_list which has the variables inside the brackets , basically the clique neighbors in order to specify to its variables their clique neighbors
                   
                   backup_list=(self.neighbors[iterate])  
                   for iterate1 in neighbors_list: #As we iterate through the variables that are in the brackets we put each variable in the dictionary clique_neighbor as a clique neighbor of the other variable , in order to have the clique neighbors.
                       if(iterate1!=iterate ):
                         
                         clique.append(iterate1)                  
                         
                   self.clique_neighbors[iterate]=clique #We fill the dictionary every time fo the specific variables that the brackets have at that time.
                   clique=[]   #When we are finished we start over with a new list with name clique , basically we remove what clique had inside so that we can use it later when we find a closing bracket for the new variablew
                neighbors_list=[] #As we did with the list clique we basically want the list neighbors_list to have its inside variables remove in order to fill it again later with new variables
                
             
                                # Basically in goal list we store for each and every variable the calculation of the clique its in .The list goal_list2 stores fo every and each variable the goal number that the clique that this variable is in needs to reach.
             if charac1=='=':   #At this part i include the goal number of every clique to a dictionary with the variables of  every clique.For example the clique [1.1,1.2]=3+ the goal number is three , so we will include it in the dictionary goal_list2  for every  variable of the clique that is in the dictionary.We will do the same with the calculation but we will add it in the goal_list dictionary. 
                 final_number=[]  #So as the reading of the kenken.text continues if we find the symbol '=' it means that the next symbol is going to be the goal number of the clique.
                 iter=0
                 num=1
                 num2=1
                 final=0
                 iter=iterator2+1
                 it=line[iter]
                 while it not in '+/-*!':  #Untill we find the calculation of the clique we put every digit of the goal number in the list final_number.We know that between '=' and the calculation of the clique , its the goal number .So we put it all in the final_number list.
                   
                     final_number.append(int(it))
                     iter=iter+1
                     it=line[iter] 
                 symbol=it   
                      
                 if len(final_number)>1: #If the fianl_number has more than one digit then we do the following

                    for num in range(1,len(final_number)): #We multiply by ten as much times as the number of the digits if we have for example the digitd 1 2 3 then we multiply by 100 to make the number 123 later
                        num2=num2*10
                    for num in final_number:#We create the goal number of the clique when we do an addition of every digits .For example we have the number 123 the above for loop will give the variable num2 the number 100 because we have three digits, so this for lopp will multiply 100 with 1 so the reasult will be 100 and after that we will divide the number num2=100 by 10 in order to have as a result 10 to multiply it by 2 to get 20 , then at the end we divide num2=10 by 10 (so we have the number 1 as a result )in order to multiply the number 1 with the number 3 and finally build the number 123                         
                        final=final +num*num2 
                        num2=num2/10 
                    final=int(final)   #The goal number                    
                    
                 if len(final_number)==1: #If we have a number with only one digit as a goal number then we just give it to final.
                     
                     final=int(final_number[0])
                     
                 for iterator3 in subgoal_list: #We put the goal number and the calculation of the clique in the dictionaries
                     
                     self.goal_list[iterator3]=symbol
                     self.goal_list2[iterator3]=final
                 subgoal_list=[]   
                 
             if charac1 in (a_list5 or a_list6) and charac2=='.' and charac3 in (a_list5 or a_list6): #We store the variables of the csp problem we see in the line of kenken.text
                                
                     number1=int(charac3)
                     number2=int(charac1)/10
                     neighbors_list.append(number1+number2) 
                     
                     subgoal_list.append(number1+number2)       
                     
               
             iterator2=iterator2+1      #A step forwards in the kenken,txt line that we read . 
             charac1=line[iterator2]
             charac2=line[iterator2-1]
             charac3=line[iterator2-2]  
             
             
                 
        CSP.__init__(self,self.variables, self.domains, self.neighbors, self.Kenken_constraints)           
                    
       
           
    def Kenken_constraints(self,A,a,B,b):        #The function for the constraints
       global counter_const 
       counter_const=counter_const+1
       
       if B in self.neighbors[A]: #if A and B are neighbors in row or column . If they have the same value then its false if not then we do the following
          
          if a==b:
                        
              return False
          if a!=b:     
              if B in self.clique_neighbors[A]: #Check If A and B are neighbors in also in clique .If they are we check the calculation of the clique.
                 storage=self.clique_neighbors[A] #storage contains the neighbors of the variable A in the same clique.
                 if self.goal_list[A]=='+':
                      final=0
                      if a+b>self.goal_list2[A]:
                        
                        return False
                      if len(storage)==1: #If we have only one neighbor in the clique so in total 2 variables then check the below
                          if a+b==self.goal_list2[A]:                        
                            
                            return True 
                      if len(storage)>1: #If we have more than one neighbor so in total 3 and above variables in the clique then do the following
                          n_list=[]
                          flag=0
                          assigns=CSP.infer_assignment(self) #We check at the moment if a clique of the variable A has a value assigned so we check for the values a and b and the value in the infer assignment if we have as a result the goal number of the clique.

                          for neighbor in self.clique_neighbors[A]: #We see how many of the clique neighbors of the A variable have a value assigned so wi collect all of them to the list n_list , in order to see if the calculation we those values has as a result the goal number of the clique.
                             
                              if neighbor in assigns and neighbor!=B:
                                  n_list.append(assigns[neighbor])
                                  
                          final=a+b  
                          if  len(n_list)!=0:  #We do the calculation of the clique with the values a,b and the assigned values to the other neihbor at that moment so we can see if the result is tho goal number of clique.
                           for i in n_list:
                              
                              final=final+i
                          
                          if final==self.goal_list2[A]: #If the reuslt is what we want then return true.Else do the basic check below
                            
 
                            return True  
                                                   
                          if final<=self.goal_list2[A]:#Here is a basic check which happens if for example we have a clique with 4 variables but at that moment some of them in the infer_assignment havent been given a value so we cant do the above.Thats why we do the below, basically if the calculation is lesser or equal to the goal number return true if its larger then return false.
                            
                            return True 
                          elif final>self.goal_list2[A]: 
                            
                            
                            return False
                          
                        
                 if self.goal_list[A]=='/':#We know that the division will have only two numbers so we do the following
                                          
                       if a/b==self.goal_list2[A] or b/a==self.goal_list2[A]:
                    
                        
                         return True
                       else :
                         
                         return False
                 if self.goal_list[A]=='*': #Multipication is exactly like addition, its basically the same except we multiply. 
                        
                        final=0
                        if a*b>self.goal_list2[A]:
                       
                          
                          return False
                        if len(storage)==1:
                          if a*b==self.goal_list2[A]:
                        
                            
                            return True 
                        
                        if len(storage)>1:
                          n_list=[]
                          flag=0
                          assigns=CSP.infer_assignment(self)
                          
                          for neighbor in self.clique_neighbors[A]:
                              if neighbor in assigns and neighbor!=B:
                                  n_list.append(assigns[neighbor])
                                  
                          final=a*b  
                          if  len(n_list)!=0:  
                           for i in n_list:
                              final=final*i
                          if final==self.goal_list2[A]:
                            
                            return True  
                          
                                         
                          if final<=self.goal_list2[A]:
                            
                            return True 
                          elif final>self.goal_list2[A]: 
                           
                           
                           return False
                 if self.goal_list[A]=='-':#We know that we can only subtract two numbers and the result must be absolute
                        
                        if abs(a-b)==self.goal_list2[A]:
                         
                          
                          return True
                        else:
                            
                            return False
                          
                            
              if B not in self.clique_neighbors[A]:#If variable B is a neighbor but not in same clique
                   
                   if self.goal_list[A]=='!': #This is the case that the clique has only one variable , so it must be the correct value to return true
                       
                      
                      if a==self.goal_list2[A]:
                        
                          
                          return True
                      if a!=self.goal_list2[A]:
                          
                          return False 
               
                   
                   return True
          
          
       if B not in self.neighbors[A] and B in self.clique_neighbors[A]:#If b is not a neighbor but it it a clique neighbor of A.We do exactly the same as above , except we don do the part that we see if the value is the a is the same with b like we used to id the variables were neighbors in row or column  .
         
                  storage=self.clique_neighbors[A]
                  if self.goal_list[A]=='+':
                      final=0
                
                      if a+b>self.goal_list2[A]:
                 
                        
                        return False
                      if len(storage)==1:
                          if a+b==self.goal_list2[A]:
                              
                        
                            
                            return True 
                      if len(storage)>1:
                          n_list=[]
                          flag=0
                          assigns=CSP.infer_assignment(self)
                          
                          for neighbor in self.clique_neighbors[A]:
                             
                              
                              if neighbor in assigns and neighbor!=B:
                                  
                                  n_list.append(assigns[neighbor])
                                  
                                
                          final=a+b  
                          if  len(n_list)!=0:  
                           for i in n_list:
                              
                              final=final+i
                          
                          if final==self.goal_list2[A]:
                            
                            
                            return True  
                           
                          
                          
                          if final<=self.goal_list2[A]:
                            
                            return True 
                          elif final>self.goal_list2[A]: 
                            
                            
                            return False
                          
                              
                         
                        
                  if self.goal_list[A]=='/':
                                          
                       if a/b==self.goal_list2[A] or b/a==self.goal_list2[A]:
                    
                         
                         return True
                       else :
                         
                         return False
                  if self.goal_list[A]=='*':
                        
                        final=0
                        if a*b>self.goal_list2[A]:
                       
                          
                          return False
                        if len(storage)==1:
                          if a*b==self.goal_list2[A]:
                        
                            
                            return True 
                        
                        if len(storage)>1:
                          n_list=[]
                          flag=0
                          assigns=CSP.infer_assignment(self)
                          
                          for neighbor in self.clique_neighbors[A]:
                             
                              
                              if neighbor in assigns and neighbor!=B:
                                  
                                  n_list.append(assigns[neighbor])
                                  
                                
                          final=a*b  
                          if  len(n_list)!=0:  
                           for i in n_list:
                              
                              final=final*i
                          
                          if final==self.goal_list2[A]:
                            
                            
                            return True  
                          
                                         
                         
                          
                          if final<=self.goal_list2[A]:
                            
                            return True 
                          elif final>self.goal_list2[A]: 
                           
                           
                           return False
                  if self.goal_list[A]=='-':
                        
                        if abs(a-b)==self.goal_list2[A]:
                         
                          
                          return True
                        else:
                            
                            return False
                         
                              
                                        
       elif B not in self.neighbors[A]:  #If b is not a neighbor of A neigther in row,column or clique
                     if self.goal_list[A]=='!':
                       if a==self.goal_list2[A]:
                          return True
                       if a!=self.goal_list[A]:
                          return False
                     return False
       

a=Kenken(1)

start=timer()
print(backtracking_search(a,select_unassigned_variable=mrv,inference=mac))
print("assigns",a.nassigns)
end=timer()
print(end-start)
print(counter_const)
#print(counter_false)
#start=timer()
#min_conflicts(a)
#print("assigns",a.nassigns)
#end=timer()
#print(end-start)

def queen_constraint(A, a, B, b):
    """Constraint is satisfied (true) if A, B are really the same variable,
    or if they are not in the same row, down diagonal, or up diagonal."""
    return A == B or (a != b and A + a != B + b and A - a != B - b)


class NQueensCSP(CSP):
    """
    Make a CSP for the nQueens problem for search with min_conflicts.
    Suitable for large n, it uses only data structures of size O(n).
    Think of placing queens one per column, from left to right.
    That means position (x, y) represents (var, val) in the CSP.
    The main structures are three arrays to count queens that could conflict:
        rows[i]      Number of queens in the ith row (i.e. val == i)
        downs[i]     Number of queens in the \ diagonal
                     such that their (x, y) coordinates sum to i
        ups[i]       Number of queens in the / diagonal
                     such that their (x, y) coordinates have x-y+n-1 = i
    We increment/decrement these counts each time a queen is placed/moved from
    a row/diagonal. So moving is O(1), as is nconflicts.  But choosing
    a variable, and a best value for the variable, are each O(n).
    If you want, you can keep track of conflicted variables, then variable
    selection will also be O(1).
    >>> len(backtracking_search(NQueensCSP(8)))
    8
    """

    def __init__(self, n):
        """Initialize data structures for n Queens."""
        CSP.__init__(self, list(range(n)), UniversalDict(list(range(n))),
                     UniversalDict(list(range(n))), queen_constraint)

        self.rows = [0] * n
        self.ups = [0] * (2 * n - 1)
        self.downs = [0] * (2 * n - 1)

    def nconflicts(self, var, val, assignment):
        """The number of conflicts, as recorded with each assignment.
        Count conflicts in row and in up, down diagonals. If there
        is a queen there, it can't conflict with itself, so subtract 3."""
        n = len(self.variables)
        c = self.rows[val] + self.downs[var + val] + self.ups[var - val + n - 1]
        if assignment.get(var, None) == val:
            c -= 3
        return c

    def assign(self, var, val, assignment):
        """Assign var, and keep track of conflicts."""
        old_val = assignment.get(var, None)
        if val != old_val:
            if old_val is not None:  # Remove old val if there was one
                self.record_conflict(assignment, var, old_val, -1)
            self.record_conflict(assignment, var, val, +1)
            CSP.assign(self, var, val, assignment)

    def unassign(self, var, assignment):
        """Remove var from assignment (if it is there) and track conflicts."""
        if var in assignment:
            self.record_conflict(assignment, var, assignment[var], -1)
        CSP.unassign(self, var, assignment)

    def record_conflict(self, assignment, var, val, delta):
        """Record conflicts caused by addition or deletion of a Queen."""
        n = len(self.variables)
        self.rows[val] += delta
        self.downs[var + val] += delta
        self.ups[var - val + n - 1] += delta

    def display(self, assignment):
        """Print the queens and the nconflicts values (for debugging)."""
        n = len(self.variables)
        for val in range(n):
            for var in range(n):
                if assignment.get(var, '') == val:
                    ch = 'Q'
                elif (var + val) % 2 == 0:
                    ch = '.'
                else:
                    ch = '-'
                print(ch, end=' ')
            print('    ', end=' ')
            for var in range(n):
                if assignment.get(var, '') == val:
                    ch = '*'
                else:
                    ch = ' '
                print(str(self.nconflicts(var, val, assignment)) + ch, end=' ')
            print()

#print(backtracking_search(NQueensCSP(8)))
# ______________________________________________________________________________
# Sudoku


def flatten(seqs):
    return sum(seqs, [])


easy1 = '..3.2.6..9..3.5..1..18.64....81.29..7.......8..67.82....26.95..8..2.3..9..5.1.3..'
harder1 = '4173698.5.3..........7......2.....6.....8.4......1.......6.3.7.5..2.....1.4......'

_R3 = list(range(3))
_CELL = itertools.count().__next__
_BGRID = [[[[_CELL() for x in _R3] for y in _R3] for bx in _R3] for by in _R3]
_BOXES = flatten([list(map(flatten, brow)) for brow in _BGRID])
_ROWS = flatten([list(map(flatten, zip(*brow))) for brow in _BGRID])
_COLS = list(zip(*_ROWS))

_NEIGHBORS = {v: set() for v in flatten(_ROWS)}
for unit in map(set, _BOXES + _ROWS + _COLS):
    for v in unit:
        _NEIGHBORS[v].update(unit - {v})


class Sudoku(CSP):
    """
    A Sudoku problem.
    The box grid is a 3x3 array of boxes, each a 3x3 array of cells.
    Each cell holds a digit in 1..9. In each box, all digits are
    different; the same for each row and column as a 9x9 grid.
    >>> e = Sudoku(easy1)
    >>> e.display(e.infer_assignment())
    . . 3 | . 2 . | 6 . .
    9 . . | 3 . 5 | . . 1
    . . 1 | 8 . 6 | 4 . .
    ------+-------+------
    . . 8 | 1 . 2 | 9 . .
    7 . . | . . . | . . 8
    . . 6 | 7 . 8 | 2 . .
    ------+-------+------
    . . 2 | 6 . 9 | 5 . .
    8 . . | 2 . 3 | . . 9
    . . 5 | . 1 . | 3 . .  
    >>> AC3(e)  # doctest: +ELLIPSIS
    (True, ...)
    >>> e.display(e.infer_assignment())
    4 8 3 | 9 2 1 | 6 5 7
    9 6 7 | 3 4 5 | 8 2 1
    2 5 1 | 8 7 6 | 4 9 3
    ------+-------+------
    5 4 8 | 1 3 2 | 9 7 6
    7 2 9 | 5 6 4 | 1 3 8
    1 3 6 | 7 9 8 | 2 4 5
    ------+-------+------
    3 7 2 | 6 8 9 | 5 1 4
    8 1 4 | 2 5 3 | 7 6 9
    6 9 5 | 4 1 7 | 3 8 2
    >>> h = Sudoku(harder1)
    >>> backtracking_search(h, select_unassigned_variable=mrv, inference=forward_checking) is not None
    True
    """

    R3 = _R3
    Cell = _CELL
    bgrid = _BGRID
    boxes = _BOXES
    rows = _ROWS
    cols = _COLS
    neighbors = _NEIGHBORS

    def __init__(self, grid):
        """Build a Sudoku problem from a string representing the grid:
        the digits 1-9 denote a filled cell, '.' or '0' an empty one;
        other characters are ignored."""
        squares = iter(re.findall(r'\d|\.', grid))
        domains = {var: [ch] if ch in '123456789' else '123456789'
                   for var, ch in zip(flatten(self.rows), squares)}
        for _ in squares:
            raise ValueError("Not a Sudoku grid", grid)  # Too many squares
        CSP.__init__(self, None, domains, self.neighbors, different_values_constraint)

    def display(self, assignment):
        def show_box(box): return [' '.join(map(show_cell, row)) for row in box]

        def show_cell(cell): return str(assignment.get(cell, '.'))

        def abut(lines1, lines2): return list(
            map(' | '.join, list(zip(lines1, lines2))))

        print('\n------+-------+------\n'.join(
            '\n'.join(reduce(
                abut, map(show_box, brow))) for brow in self.bgrid))


# ______________________________________________________________________________
# The Zebra Puzzle


def Zebra():
    """Return an instance of the Zebra Puzzle."""
    Colors = 'Red Yellow Blue Green Ivory'.split()
    Pets = 'Dog Fox Snails Horse Zebra'.split()
    Drinks = 'OJ Tea Coffee Milk Water'.split()
    Countries = 'Englishman Spaniard Norwegian Ukranian Japanese'.split()
    Smokes = 'Kools Chesterfields Winston LuckyStrike Parliaments'.split()
    variables = Colors + Pets + Drinks + Countries + Smokes
    domains = {}
    for var in variables:
        domains[var] = list(range(1, 6))
    domains['Norwegian'] = [1]
    domains['Milk'] = [3]
    neighbors = parse_neighbors("""Englishman: Red;
                Spaniard: Dog; Kools: Yellow; Chesterfields: Fox;
                Norwegian: Blue; Winston: Snails; LuckyStrike: OJ;
                Ukranian: Tea; Japanese: Parliaments; Kools: Horse;
                Coffee: Green; Green: Ivory""")
    for type in [Colors, Pets, Drinks, Countries, Smokes]:
        for A in type:
            for B in type:
                if A != B:
                    if B not in neighbors[A]:
                        neighbors[A].append(B)
                    if A not in neighbors[B]:
                        neighbors[B].append(A)

    def zebra_constraint(A, a, B, b, recurse=0):
        same = (a == b)
        next_to = abs(a - b) == 1
        if A == 'Englishman' and B == 'Red':
            return same
        if A == 'Spaniard' and B == 'Dog':
            return same
        if A == 'Chesterfields' and B == 'Fox':
            return next_to
        if A == 'Norwegian' and B == 'Blue':
            return next_to
        if A == 'Kools' and B == 'Yellow':
            return same
        if A == 'Winston' and B == 'Snails':
            return same
        if A == 'LuckyStrike' and B == 'OJ':
            return same
        if A == 'Ukranian' and B == 'Tea':
            return same
        if A == 'Japanese' and B == 'Parliaments':
            return same
        if A == 'Kools' and B == 'Horse':
            return next_to
        if A == 'Coffee' and B == 'Green':
            return same
        if A == 'Green' and B == 'Ivory':
            return a - 1 == b
        if recurse == 0:
            return zebra_constraint(B, b, A, a, 1)
        if ((A in Colors and B in Colors) or
                (A in Pets and B in Pets) or
                (A in Drinks and B in Drinks) or
                (A in Countries and B in Countries) or
                (A in Smokes and B in Smokes)):
            return not same
        raise Exception('error')

    return CSP(variables, domains, neighbors, zebra_constraint)


def solve_zebra(algorithm=min_conflicts, **args):
    z = Zebra()
    ans = algorithm(z, **args)
    for h in range(1, 6):
        print('House', h, end=' ')
        for (var, val) in ans.items():
            if val == h:
                print(var, end=' ')
        print()
    return ans['Zebra'], ans['Water'], z.nassigns, ans


# ______________________________________________________________________________
# n-ary Constraint Satisfaction Problem

class NaryCSP:
    """
    A nary-CSP consists of:
    domains     : a dictionary that maps each variable to its domain
    constraints : a list of constraints
    variables   : a set of variables
    var_to_const: a variable to set of constraints dictionary
    """

    def __init__(self, domains, constraints):
        """Domains is a variable:domain dictionary
        constraints is a list of constraints
        """
        self.variables = set(domains)
        self.domains = domains
        self.constraints = constraints
        self.var_to_const = {var: set() for var in self.variables}
        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)

    def __str__(self):
        """String representation of CSP"""
        return str(self.domains)

    def display(self, assignment=None):
        """More detailed string representation of CSP"""
        if assignment is None:
            assignment = {}
        print(assignment)

    def consistent(self, assignment):
        """assignment is a variable:value dictionary
        returns True if all of the constraints that can be evaluated
                        evaluate to True given assignment.
        """
        return all(con.holds(assignment)
                   for con in self.constraints
                   if all(v in assignment for v in con.scope))


class Constraint:
    """
    A Constraint consists of:
    scope    : a tuple of variables
    condition: a function that can applied to a tuple of values
    for the variables.
    """

    def __init__(self, scope, condition):
        self.scope = scope
        self.condition = condition

    def __repr__(self):
        return self.condition.__name__ + str(self.scope)

    def holds(self, assignment):
        """Returns the value of Constraint con evaluated in assignment.

        precondition: all variables are assigned in assignment
        """
        return self.condition(*tuple(assignment[v] for v in self.scope))


def all_diff_constraint(*values):
    """Returns True if all values are different, False otherwise"""
    return len(values) is len(set(values))


def is_word_constraint(words):
    """Returns True if the letters concatenated form a word in words, False otherwise"""

    def isw(*letters):
        return "".join(letters) in words

    return isw


def meet_at_constraint(p1, p2):
    """Returns a function that is True when the words meet at the positions (p1, p2), False otherwise"""

    def meets(w1, w2):
        return w1[p1] == w2[p2]

    meets.__name__ = "meet_at(" + str(p1) + ',' + str(p2) + ')'
    return meets


def adjacent_constraint(x, y):
    """Returns True if x and y are adjacent numbers, False otherwise"""
    return abs(x - y) == 1


def sum_constraint(n):
    """Returns a function that is True when the the sum of all values is n, False otherwise"""

    def sumv(*values):
        return sum(values) is n

    sumv.__name__ = str(n) + "==sum"
    return sumv


def is_constraint(val):
    """Returns a function that is True when x is equal to val, False otherwise"""

    def isv(x):
        return val == x

    isv.__name__ = str(val) + "=="
    return isv


def ne_constraint(val):
    """Returns a function that is True when x is not equal to val, False otherwise"""

    def nev(x):
        return val != x

    nev.__name__ = str(val) + "!="
    return nev


def no_heuristic(to_do):
    return to_do


def sat_up(to_do):
    return SortedSet(to_do, key=lambda t: 1 / len([var for var in t[1].scope]))


class ACSolver:
    """Solves a CSP with arc consistency and domain splitting"""

    def __init__(self, csp):
        """a CSP solver that uses arc consistency
        * csp is the CSP to be solved
        """
        self.csp = csp

    def GAC(self, orig_domains=None, to_do=None, arc_heuristic=sat_up):
        """
        Makes this CSP arc-consistent using Generalized Arc Consistency
        orig_domains: is the original domains
        to_do       : is a set of (variable,constraint) pairs
        returns the reduced domains (an arc-consistent variable:domain dictionary)
        """
        if orig_domains is None:
            orig_domains = self.csp.domains
        if to_do is None:
            to_do = {(var, const) for const in self.csp.constraints for var in const.scope}
        else:
            to_do = to_do.copy()
        domains = orig_domains.copy()
        to_do = arc_heuristic(to_do)
        checks = 0
        while to_do:
            var, const = to_do.pop()
            other_vars = [ov for ov in const.scope if ov != var]
            new_domain = set()
            if len(other_vars) == 0:
                for val in domains[var]:
                    if const.holds({var: val}):
                        new_domain.add(val)
                    checks += 1
                # new_domain = {val for val in domains[var]
                #               if const.holds({var: val})}
            elif len(other_vars) == 1:
                other = other_vars[0]
                for val in domains[var]:
                    for other_val in domains[other]:
                        checks += 1
                        if const.holds({var: val, other: other_val}):
                            new_domain.add(val)
                            break
                # new_domain = {val for val in domains[var]
                #               if any(const.holds({var: val, other: other_val})
                #                      for other_val in domains[other])}
            else:  # general case
                for val in domains[var]:
                    holds, checks = self.any_holds(domains, const, {var: val}, other_vars, checks=checks)
                    if holds:
                        new_domain.add(val)
                # new_domain = {val for val in domains[var]
                #               if self.any_holds(domains, const, {var: val}, other_vars)}
            if new_domain != domains[var]:
                domains[var] = new_domain
                if not new_domain:
                    return False, domains, checks
                add_to_do = self.new_to_do(var, const).difference(to_do)
                to_do |= add_to_do
        return True, domains, checks

    def new_to_do(self, var, const):
        """
        Returns new elements to be added to to_do after assigning
        variable var in constraint const.
        """
        return {(nvar, nconst) for nconst in self.csp.var_to_const[var]
                if nconst != const
                for nvar in nconst.scope
                if nvar != var}

    def any_holds(self, domains, const, env, other_vars, ind=0, checks=0):
        """
        Returns True if Constraint const holds for an assignment
        that extends env with the variables in other_vars[ind:]
        env is a dictionary
        Warning: this has side effects and changes the elements of env
        """
        if ind == len(other_vars):
            return const.holds(env), checks + 1
        else:
            var = other_vars[ind]
            for val in domains[var]:
                # env = dict_union(env, {var:val})  # no side effects
                env[var] = val
                holds, checks = self.any_holds(domains, const, env, other_vars, ind + 1, checks)
                if holds:
                    return True, checks
            return False, checks

    def domain_splitting(self, domains=None, to_do=None, arc_heuristic=sat_up):
        """
        Return a solution to the current CSP or False if there are no solutions
        to_do is the list of arcs to check
        """
        if domains is None:
            domains = self.csp.domains
        consistency, new_domains, _ = self.GAC(domains, to_do, arc_heuristic)
        if not consistency:
            return False
        elif all(len(new_domains[var]) == 1 for var in domains):
            return {var: first(new_domains[var]) for var in domains}
        else:
            var = first(x for x in self.csp.variables if len(new_domains[x]) > 1)
            if var:
                dom1, dom2 = partition_domain(new_domains[var])
                new_doms1 = extend(new_domains, var, dom1)
                new_doms2 = extend(new_domains, var, dom2)
                to_do = self.new_to_do(var, None)
                return self.domain_splitting(new_doms1, to_do, arc_heuristic) or \
                       self.domain_splitting(new_doms2, to_do, arc_heuristic)


def partition_domain(dom):
    """Partitions domain dom into two"""
    split = len(dom) // 2
    dom1 = set(list(dom)[:split])
    dom2 = dom - dom1
    return dom1, dom2


class ACSearchSolver(search.Problem):
    """A search problem with arc consistency and domain splitting
    A node is a CSP"""

    def __init__(self, csp, arc_heuristic=sat_up):
        self.cons = ACSolver(csp)
        consistency, self.domains, _ = self.cons.GAC(arc_heuristic=arc_heuristic)
        if not consistency:
            raise Exception('CSP is inconsistent')
        self.heuristic = arc_heuristic
        super().__init__(self.domains)

    def goal_test(self, node):
        """Node is a goal if all domains have 1 element"""
        return all(len(node[var]) == 1 for var in node)

    def actions(self, state):
        var = first(x for x in state if len(state[x]) > 1)
        neighs = []
        if var:
            dom1, dom2 = partition_domain(state[var])
            to_do = self.cons.new_to_do(var, None)
            for dom in [dom1, dom2]:
                new_domains = extend(state, var, dom)
                consistency, cons_doms, _ = self.cons.GAC(new_domains, to_do, self.heuristic)
                if consistency:
                    neighs.append(cons_doms)
        return neighs

    def result(self, state, action):
        return action


def ac_solver(csp, arc_heuristic=sat_up):
    """Arc consistency (domain splitting interface)"""
    return ACSolver(csp).domain_splitting(arc_heuristic=arc_heuristic)


def ac_search_solver(csp, arc_heuristic=sat_up):
    """Arc consistency (search interface)"""
    from search import depth_first_tree_search
    solution = None
    try:
        solution = depth_first_tree_search(ACSearchSolver(csp, arc_heuristic=arc_heuristic)).state
    except:
        return solution
    if solution:
        return {var: first(solution[var]) for var in solution}


# ______________________________________________________________________________
# Crossword Problem


csp_crossword = NaryCSP({'one_across': {'ant', 'big', 'bus', 'car', 'has'},
                         'one_down': {'book', 'buys', 'hold', 'lane', 'year'},
                         'two_down': {'ginger', 'search', 'symbol', 'syntax'},
                         'three_across': {'book', 'buys', 'hold', 'land', 'year'},
                         'four_across': {'ant', 'big', 'bus', 'car', 'has'}},
                        [Constraint(('one_across', 'one_down'), meet_at_constraint(0, 0)),
                         Constraint(('one_across', 'two_down'), meet_at_constraint(2, 0)),
                         Constraint(('three_across', 'two_down'), meet_at_constraint(2, 2)),
                         Constraint(('three_across', 'one_down'), meet_at_constraint(0, 2)),
                         Constraint(('four_across', 'two_down'), meet_at_constraint(0, 4))])

crossword1 = [['_', '_', '_', '*', '*'],
              ['_', '*', '_', '*', '*'],
              ['_', '_', '_', '_', '*'],
              ['_', '*', '_', '*', '*'],
              ['*', '*', '_', '_', '_'],
              ['*', '*', '_', '*', '*']]

words1 = {'ant', 'big', 'bus', 'car', 'has', 'book', 'buys', 'hold',
          'lane', 'year', 'ginger', 'search', 'symbol', 'syntax'}


class Crossword(NaryCSP):

    def __init__(self, puzzle, words):
        domains = {}
        constraints = []
        for i, line in enumerate(puzzle):
            scope = []
            for j, element in enumerate(line):
                if element == '_':
                    var = "p" + str(j) + str(i)
                    domains[var] = list(string.ascii_lowercase)
                    scope.append(var)
                else:
                    if len(scope) > 1:
                        constraints.append(Constraint(tuple(scope), is_word_constraint(words)))
                    scope.clear()
            if len(scope) > 1:
                constraints.append(Constraint(tuple(scope), is_word_constraint(words)))
        puzzle_t = list(map(list, zip(*puzzle)))
        for i, line in enumerate(puzzle_t):
            scope = []
            for j, element in enumerate(line):
                if element == '_':
                    scope.append("p" + str(i) + str(j))
                else:
                    if len(scope) > 1:
                        constraints.append(Constraint(tuple(scope), is_word_constraint(words)))
                    scope.clear()
            if len(scope) > 1:
                constraints.append(Constraint(tuple(scope), is_word_constraint(words)))
        super().__init__(domains, constraints)
        self.puzzle = puzzle

    def display(self, assignment=None):
        for i, line in enumerate(self.puzzle):
            puzzle = ""
            for j, element in enumerate(line):
                if element == '*':
                    puzzle += "[*] "
                else:
                    var = "p" + str(j) + str(i)
                    if assignment is not None:
                        if isinstance(assignment[var], set) and len(assignment[var]) == 1:
                            puzzle += "[" + str(first(assignment[var])).upper() + "] "
                        elif isinstance(assignment[var], str):
                            puzzle += "[" + str(assignment[var]).upper() + "] "
                        else:
                            puzzle += "[_] "
                    else:
                        puzzle += "[_] "
            print(puzzle)


# ______________________________________________________________________________
# Kakuro Problem


# difficulty 0
kakuro1 = [['*', '*', '*', [6, ''], [3, '']],
           ['*', [4, ''], [3, 3], '_', '_'],
           [['', 10], '_', '_', '_', '_'],
           [['', 3], '_', '_', '*', '*']]

# difficulty 0
kakuro2 = [
    ['*', [10, ''], [13, ''], '*'],
    [['', 3], '_', '_', [13, '']],
    [['', 12], '_', '_', '_'],
    [['', 21], '_', '_', '_']]

# difficulty 1
kakuro3 = [
    ['*', [17, ''], [28, ''], '*', [42, ''], [22, '']],
    [['', 9], '_', '_', [31, 14], '_', '_'],
    [['', 20], '_', '_', '_', '_', '_'],
    ['*', ['', 30], '_', '_', '_', '_'],
    ['*', [22, 24], '_', '_', '_', '*'],
    [['', 25], '_', '_', '_', '_', [11, '']],
    [['', 20], '_', '_', '_', '_', '_'],
    [['', 14], '_', '_', ['', 17], '_', '_']]

# difficulty 2
kakuro4 = [
    ['*', '*', '*', '*', '*', [4, ''], [24, ''], [11, ''], '*', '*', '*', [11, ''], [17, ''], '*', '*'],
    ['*', '*', '*', [17, ''], [11, 12], '_', '_', '_', '*', '*', [24, 10], '_', '_', [11, ''], '*'],
    ['*', [4, ''], [16, 26], '_', '_', '_', '_', '_', '*', ['', 20], '_', '_', '_', '_', [16, '']],
    [['', 20], '_', '_', '_', '_', [24, 13], '_', '_', [16, ''], ['', 12], '_', '_', [23, 10], '_', '_'],
    [['', 10], '_', '_', [24, 12], '_', '_', [16, 5], '_', '_', [16, 30], '_', '_', '_', '_', '_'],
    ['*', '*', [3, 26], '_', '_', '_', '_', ['', 12], '_', '_', [4, ''], [16, 14], '_', '_', '*'],
    ['*', ['', 8], '_', '_', ['', 15], '_', '_', [34, 26], '_', '_', '_', '_', '_', '*', '*'],
    ['*', ['', 11], '_', '_', [3, ''], [17, ''], ['', 14], '_', '_', ['', 8], '_', '_', [7, ''], [17, ''], '*'],
    ['*', '*', '*', [23, 10], '_', '_', [3, 9], '_', '_', [4, ''], [23, ''], ['', 13], '_', '_', '*'],
    ['*', '*', [10, 26], '_', '_', '_', '_', '_', ['', 7], '_', '_', [30, 9], '_', '_', '*'],
    ['*', [17, 11], '_', '_', [11, ''], [24, 8], '_', '_', [11, 21], '_', '_', '_', '_', [16, ''], [17, '']],
    [['', 29], '_', '_', '_', '_', '_', ['', 7], '_', '_', [23, 14], '_', '_', [3, 17], '_', '_'],
    [['', 10], '_', '_', [3, 10], '_', '_', '*', ['', 8], '_', '_', [4, 25], '_', '_', '_', '_'],
    ['*', ['', 16], '_', '_', '_', '_', '*', ['', 23], '_', '_', '_', '_', '_', '*', '*'],
    ['*', '*', ['', 6], '_', '_', '*', '*', ['', 15], '_', '_', '_', '*', '*', '*', '*']]


class Kakuro(NaryCSP):

    def __init__(self, puzzle):
        variables = []
        for i, line in enumerate(puzzle):
            # print line
            for j, element in enumerate(line):
                if element == '_':
                    var1 = str(i)
                    if len(var1) == 1:
                        var1 = "0" + var1
                    var2 = str(j)
                    if len(var2) == 1:
                        var2 = "0" + var2
                    variables.append("X" + var1 + var2)
        domains = {}
        for var in variables:
            domains[var] = set(range(1, 10))
        constraints = []
        for i, line in enumerate(puzzle):
            for j, element in enumerate(line):
                if element != '_' and element != '*':
                    # down - column
                    if element[0] != '':
                        x = []
                        for k in range(i + 1, len(puzzle)):
                            if puzzle[k][j] != '_':
                                break
                            var1 = str(k)
                            if len(var1) == 1:
                                var1 = "0" + var1
                            var2 = str(j)
                            if len(var2) == 1:
                                var2 = "0" + var2
                            x.append("X" + var1 + var2)
                        constraints.append(Constraint(x, sum_constraint(element[0])))
                        constraints.append(Constraint(x, all_diff_constraint))
                    # right - line
                    if element[1] != '':
                        x = []
                        for k in range(j + 1, len(puzzle[i])):
                            if puzzle[i][k] != '_':
                                break
                            var1 = str(i)
                            if len(var1) == 1:
                                var1 = "0" + var1
                            var2 = str(k)
                            if len(var2) == 1:
                                var2 = "0" + var2
                            x.append("X" + var1 + var2)
                        constraints.append(Constraint(x, sum_constraint(element[1])))
                        constraints.append(Constraint(x, all_diff_constraint))
        super().__init__(domains, constraints)
        self.puzzle = puzzle

    def display(self, assignment=None):
        for i, line in enumerate(self.puzzle):
            puzzle = ""
            for j, element in enumerate(line):
                if element == '*':
                    puzzle += "[*]\t"
                elif element == '_':
                    var1 = str(i)
                    if len(var1) == 1:
                        var1 = "0" + var1
                    var2 = str(j)
                    if len(var2) == 1:
                        var2 = "0" + var2
                    var = "X" + var1 + var2
                    if assignment is not None:
                        if isinstance(assignment[var], set) and len(assignment[var]) == 1:
                            puzzle += "[" + str(first(assignment[var])) + "]\t"
                        elif isinstance(assignment[var], int):
                            puzzle += "[" + str(assignment[var]) + "]\t"
                        else:
                            puzzle += "[_]\t"
                    else:
                        puzzle += "[_]\t"
                else:
                    puzzle += str(element[0]) + "\\" + str(element[1]) + "\t"
            print(puzzle)


# ______________________________________________________________________________
# Cryptarithmetic Problem

# [Figure 6.2]
# T W O + T W O = F O U R
two_two_four = NaryCSP({'T': set(range(1, 10)), 'F': set(range(1, 10)),
                        'W': set(range(0, 10)), 'O': set(range(0, 10)), 'U': set(range(0, 10)), 'R': set(range(0, 10)),
                        'C1': set(range(0, 2)), 'C2': set(range(0, 2)), 'C3': set(range(0, 2))},
                       [Constraint(('T', 'F', 'W', 'O', 'U', 'R'), all_diff_constraint),
                        Constraint(('O', 'R', 'C1'), lambda o, r, c1: o + o == r + 10 * c1),
                        Constraint(('W', 'U', 'C1', 'C2'), lambda w, u, c1, c2: c1 + w + w == u + 10 * c2),
                        Constraint(('T', 'O', 'C2', 'C3'), lambda t, o, c2, c3: c2 + t + t == o + 10 * c3),
                        Constraint(('F', 'C3'), eq)])

# S E N D + M O R E = M O N E Y
send_more_money = NaryCSP({'S': set(range(1, 10)), 'M': set(range(1, 10)),
                           'E': set(range(0, 10)), 'N': set(range(0, 10)), 'D': set(range(0, 10)),
                           'O': set(range(0, 10)), 'R': set(range(0, 10)), 'Y': set(range(0, 10)),
                           'C1': set(range(0, 2)), 'C2': set(range(0, 2)), 'C3': set(range(0, 2)),
                           'C4': set(range(0, 2))},
                          [Constraint(('S', 'E', 'N', 'D', 'M', 'O', 'R', 'Y'), all_diff_constraint),
                           Constraint(('D', 'E', 'Y', 'C1'), lambda d, e, y, c1: d + e == y + 10 * c1),
                           Constraint(('N', 'R', 'E', 'C1', 'C2'), lambda n, r, e, c1, c2: c1 + n + r == e + 10 * c2),
                           Constraint(('E', 'O', 'N', 'C2', 'C3'), lambda e, o, n, c2, c3: c2 + e + o == n + 10 * c3),
                           Constraint(('S', 'M', 'O', 'C3', 'C4'), lambda s, m, o, c3, c4: c3 + s + m == o + 10 * c4),
                           Constraint(('M', 'C4'), eq)])


#e = Sudoku(easy1)
#print(AC3(e))

#h = Sudoku(harder1)
#backtracki