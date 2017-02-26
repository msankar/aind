"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import sys

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

def keepYourDistance(game, player):
    """
        Keep your distance from opponent. Reward larger difference in distance
        between the location vectors of maximizing and minimizing players with higher score.

        Note: this function should be called from within a Player instance as
        `self.score()` -- you should not need to call this function directly.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The heuristic value of the current game state to the specified player.
        """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    oppLoc = game.get_player_location(game.get_opponent(player))
    if oppLoc == None:
        return float(0)

    myLoc = game.get_player_location(player)
    if myLoc == None:
        return float(0)

    return float(abs(sum(oppLoc) - sum(myLoc)))
#__________________________________________________________________________

def keepYourEnemiesCloser(game, player):
    """
    Keep your opponent close. Reward smaller difference in distance
    between the location vectors of minimizing and maximizing players
    with a higher score.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    oppLoc = game.get_player_location(game.get_opponent(player))
    if oppLoc == None:
        return 0.

    myLoc = game.get_player_location(player)
    if myLoc == None:
        return 0.

    return float(-abs(sum(oppLoc) - sum(myLoc)))
#__________________________________________________________________________

def numberOfOpponentVsMyMoves(game, player):
    """
    This score function returns the difference
    between the number of moves available for self and the opponent player.
    Add a weighted factor to the sum of own moves and the opponents moves.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    weight = 0.45

    ownMoves = len(game.get_legal_moves(player))
    oppMoves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(weight * ownMoves + (1 - weight) * (-oppMoves))
#__________________________________________________________________________

def forecastAndLookAhead(game, player):
    """ Look ahead moves for both self and opponent

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    ownMoves = game.get_legal_moves(player)
    oppMoves = game.get_legal_moves(game.get_opponent(player))

    weight = 0.5

    if len(ownMoves) == 0:
        return float("-inf")

    lookAheadMyMoves = 0.0
    lookaheadOppMoves = 0.0

    for move in ownMoves:
        # Look ahead self moves and add it to existing moves.
        lookAheadMyMoves += len(game.forecast_move(move).get_legal_moves(player))
        # Look ahead opp moves at the next level.
        lookaheadOppMoves += len(game.forecast_move(move).get_legal_moves(game.get_opponent(player)))
    # Average moves self player has as a % of my moves
    lookAheadMyMoves = lookAheadMyMoves/len(ownMoves)
    # Average moves opponent has as a % of my moves
    lookaheadOppMoves = lookaheadOppMoves/len(ownMoves)

    if lookAheadMyMoves == 0:
        pass

    score = weight * (len(ownMoves) - len(oppMoves)) + (1 - weight) * (lookAheadMyMoves - lookaheadOppMoves)
    return float(score)
#__________________________________________________________________________

def getOutOfCorner(game, player):
    """
    For the remaining moves, penalize corner moves for self and
    reward corner moves for opponent. If the number of remaining blank
    space is about 20% of the board increase the penalty/reward factor.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    penaltyRewardFactor = 1
    # If you are in the corner as the game is closer to the end penalize heavily
    if len(game.get_blank_spaces()) < (game.width * game.height / 5.):
        penaltyRewardFactor = 5

    # corners
    corners = [(0, 0),(0, (game.width - 1)),
               ((game.height - 1), 0),((game.height - 1), (game.width - 1))]

    ownMoves = game.get_legal_moves(player)
    selfInCorner = [move for move in ownMoves if move in corners]
    oppMoves = game.get_legal_moves(game.get_opponent(player))
    oppInCorner = [move for move in oppMoves if move in corners]

    #Penalize self for being in the corner, reward opponent for being in the corner
    return float(len(ownMoves) - (penaltyRewardFactor * len(selfInCorner))
                 - len(oppMoves) + (penaltyRewardFactor * len(oppInCorner)))
#__________________________________________________________________________

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # return keepYourDistance(game, player)
    # return keepYourEnemiesCloser(game, player)
    # return forecastAndLookAhead(game, player)
    # return getOutOfCorner(game, player)
    return numberOfOpponentVsMyMoves(game, player)
#__________________________________________________________________________

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        if not legal_moves:
            return (-1, -1)

        bestMove = None

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative:
                if self.method == "minimax":
                    for depth in range(sys.maxsize):
                        _, move = self.minimax(game, depth)
                        bestMove = move
                if self.method == "alphabeta":
                    for depth in range(sys.maxsize):
                        _, move = self.alphabeta(game, depth)
                        bestMove = move
            else:
                if self.method == "minimax":
                    _, bestMove = self.minimax(game, self.search_depth)
                if self.method == "alphabeta":
                    _, bestMove = self.alphabeta(game, self.search_depth)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return bestMove

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Get a list of all available legal moves for current player.
        legalMoves = game.get_legal_moves()

        # If there are no more legal moves return tuple (-1, -1). We are done.
        if not legalMoves:
            return game.utility(self), (-1, -1)

        # If depth is 0, then we have traversed the desired depth of the tree
        # Return the score.
        if depth == 0:
            return self.score(game, self), (-1, -1)

        # Legal moves exist and we have not yet traversed the target depth of the tree.
        # Recursively call minimax and return best possible branch at each level.
        # Remember, for MAXIMIZING player it is the branch with HIGHEST score.
        # For MINIMIZING player, it is the branch with LOWEST score.
        bestMove = None
        if maximizing_player:
            # For the maximizing player, best score is the HIGHEST score.
            bestScore = float("-inf")
            for move in legalMoves:
                # Recursively call minimax until we traverse the desired depth of the tree.
                score, _ = self.minimax(game.forecast_move(move), depth - 1, False)
                if score > bestScore:
                    bestScore, bestMove = score, move
        else: # This is a minimizing player.
            # For a minimizing player best score is the LOWEST score.
            bestScore = float("inf")
            for move in legalMoves:
                # Recursively call minimax until we traverse the desired depth of the tree.
                score, _ = self.minimax(game.forecast_move(move), depth - 1, True)
                if score < bestScore:
                    bestScore, bestMove = score, move
        return bestScore, bestMove

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        # Find all legal moves for the player.
        legalMoves = game.get_legal_moves()

        # If there are no legal moves return tuple (-1, -1)
        if not legalMoves:
            return game.utility(self), (-1, -1)

        # If depth is 0, then we have traversed the desired depth of the tree
        # Return the score.
        if depth == 0:
            return self.score(game, self), (-1, -1)

        bestMove = None
        if maximizing_player:
            # HIGHEST score is the best score for maximizing player.
            bestScore = float("-inf")
            for move in legalMoves:
                # forecast next move for current player and recursively call alphabeta.
                score, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, False)
                if score > bestScore:
                    bestScore, bestMove = score, move
                # Prune.
                # Beta is the lowest score so far (+inf is the worst)
                if bestScore >= beta:
                    return bestScore, bestMove
                # Alpha is the highest score so far.
                alpha = max(alpha, bestScore)
        else: # Else minimizing player
            # LOWEST score is best for minimizing player.
            bestScore = float("inf")
            for move in legalMoves:
                # forecast next move for current player and recursively call alphabeta
                score, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, True)
                if score < bestScore:
                    bestScore, bestMove = score, move
                # Prune
                # Alpha is the highest score so far (-inf is the worst)
                if bestScore <= alpha:
                    return bestScore, bestMove
                # Beta is the lowest score so far.
                beta = min(beta, bestScore)
        return bestScore, bestMove
