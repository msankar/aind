# Artificial Intelligence Nanodegree
## Introductory Project: Diagonal Sudoku Solver

# Question 1 (Naked Twins)
Q: How do we use constraint propagation to solve the naked twins problem?  
A: Constraint propagation a form of reasoning, using a network of related facts, in which a value or range of possible values determined for one variable constraints the possible values of variables to which it is related. (www.cs.utexas.edu/~novak/aivocab.html).

Here we are applying a constraint repeatedly until we arrive at a solution. Naked twins strategy in Sudoku identifies all boxes that have 2 elements. We then pair up boxes that have the same two elements (twins). Finally we remove these 2 numbers from the peers of each of the twin boxes. See naked_twins function in solution.py along with its usage in reduce_puzzle.


# Question 2 (Diagonal Sudoku)
Q: How do we use constraint propagation to solve the diagonal sudoku problem?  
A: Diagonal constraint is added as another unit for diagonal sudoku, all boxes on the diagonal will have its diagonal entries as its peers, in additions to its row, column and square units. This forces in only accepting a solution that satisfies this constraint.

	diag1_units = [[rows[i]+cols[i] for i in range(len(rows))]]
	diag2_units = [[rows[i]+cols[::-1][i] for i in range(len(rows))]]
	unitlist = row_units + column_units + square_units + diag1_units + diag2_units

### Install

This project requires **Python 3**.

We recommend students install [Anaconda](https://www.continuum.io/downloads), a pre-packaged Python distribution that contains all of the necessary libraries and software for this project. 
Please try using the environment we provided in the Anaconda lesson of the Nanodegree.

##### Optional: Pygame

Optionally, you can also install pygame if you want to see your visualization. If you've followed our instructions for setting up our conda environment, you should be all set.

If not, please see how to download pygame [here](http://www.pygame.org/download.shtml).

### Code

* `solutions.py` - You'll fill this in as part of your solution.
* `solution_test.py` - Do not modify this. You can test your solution by running `python solution_test.py`.
* `PySudoku.py` - Do not modify this. This is code for visualizing your solution.
* `visualize.py` - Do not modify this. This is code for visualizing your solution.

### Visualizing

To visualize your solution, please only assign values to the values_dict using the ```assign_values``` function provided in solution.py

### Data

The data consists of a text file of diagonal sudokus for you to solve.