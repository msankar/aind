# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
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
     Returns the start state for the search problem 
     """
     util.raiseNotDefined()
    
  def isGoalState(self, state):
     """
       state: Search state
    
     Returns True if and only if the state is a valid goal state
     """
     util.raiseNotDefined()

  def getSuccessors(self, state):
     """
       state: Search state
     
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
     """
     util.raiseNotDefined()

  def getCostOfActions(self, actions):
     """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
     """
     util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first
  [2nd Edition: p 75, 3rd Edition: p 87]
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm 
  [2nd Edition: Fig. 3.18, 3rd Edition: Fig 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())
  """
  print "Start:", problem.getStartState()
  print "Is the start a goal?", problem.isGoalState(problem.getStartState())
  print "Start's successors:", problem.getSuccessors(problem.getStartState())

  fringe = util.Stack() # LIFO
  # push the starting node
  fringe.push((problem.getStartState(), []))
  visited = []

  while not fringe.isEmpty():
      currentNode, directionsList = fringe.pop()
      #For each one of the successor node. move the fringe and add to visited.
      for successorNode, direction, steps in problem.getSuccessors(currentNode):
          if not successorNode in visited:
              if problem.isGoalState(successorNode):
                  return directionsList + [direction]
              fringe.push((successorNode, directionsList + [direction]))
              visited.append(currentNode)

  return []

  util.raiseNotDefined()

def breadthFirstSearch(problem):
    """
    Search the shallowest nodes in the search tree first.
    [2nd Edition: p 73, 3rd Edition: p 82]
    """
    fringe = util.Queue() #FIFO
    fringe.push( (problem.getStartState(), []) )
    visited = []

    while not fringe.isEmpty():
        currentNode, directionsList = fringe.pop()

        for successorNode, dir, steps in problem.getSuccessors(currentNode):
            if not successorNode in visited:
                if problem.isGoalState(successorNode):
                    return directionsList + [dir]
                fringe.push((successorNode, directionsList + [dir]))
                visited.append(successorNode)

    return []
    util.raiseNotDefined()
      
def uniformCostSearch(problem):
  "Search the node of least total cost first. "
  fringe = util.PriorityQueue()

  # Push the starting node with lowest priority
  fringe.push((problem.getStartState(), []), 0) # item (node and directions) and priority
  explored = []

  while not fringe.isEmpty():
      currentNode, directions = fringe.pop() # returns item

      # If current node is goal return.
      if problem.isGoalState(currentNode):
          return directions

      # Current node is explored.
      explored.append(currentNode)

      # For each one of the successor nodes add to the fringe with its cost.
      for successorNode, direction, steps in problem.getSuccessors(currentNode):
          if not successorNode in explored:
              new_actions = directions + [direction]
              fringe.push((successorNode, new_actions), problem.getCostOfActions(new_actions))

  return []
  util.raiseNotDefined()

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  "Search the node that has the lowest combined cost and heuristic first."
  fringe = util.PriorityQueue()

  # Heuristics take two arguments: a state in the search problem (the main argument),
  # and the problem itself (for reference information).
  fringe.push((problem.getStartState(), []), heuristic(problem.getStartState(), problem))

  explored = []

  while not fringe.isEmpty():
      currentNode, actions = fringe.pop()

      # If current node is goal return
      if problem.isGoalState(currentNode):
          return actions

      # Add current node to explored.
      explored.append(currentNode)

      for successiveNode, direction, cost in problem.getSuccessors(currentNode):
          if not successiveNode in explored:
              new_actions = actions + [direction]
              # f = (g+h) path cost + estimated distance to the goal.
              minFValue = problem.getCostOfActions(new_actions) + heuristic(successiveNode, problem)
              fringe.push((successiveNode, new_actions), minFValue)

  return []
  util.raiseNotDefined()
    
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
