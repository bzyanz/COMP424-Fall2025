# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
from helpers import get_valid_moves, execute_move, check_endgame

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    CORRECT α-β pruning implementation following COMP 424 pseudocode
    """
    
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "CorrectAlphaBeta"
        self.nodes_evaluated = 0
    
    def step(self, chess_board, player, opponent):
        """
        Depth-limited α-β search
        """
        start_time = time.time()
        self.nodes_evaluated = 0
        
        moves = get_valid_moves(chess_board, player)
        if not moves:
            return None
        
        # Order moves for better pruning
        moves = self.order_moves(chess_board, moves, player)
        
        # α-β search with depth 3
        depth = 3
        best_move = None
        best_value = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for move in moves:
            # Try this move
            new_board = deepcopy(chess_board)
            execute_move(new_board, move, player)
            
            # Min's turn (opponent)
            value = self.min_value(new_board, depth-1, player, opponent, alpha, beta)
            
            if value > best_value:
                best_value = value
                best_move = move
            
            alpha = max(alpha, best_value)
        
        time_taken = time.time() - start_time
        print(f"α-β: depth={depth}, nodes={self.nodes_evaluated}, time={time_taken:.3f}s, value={best_value}")
        
        return best_move if best_move else moves[0]
    
    def max_value(self, state, depth, player, opponent, alpha, beta):
        """
        Max player's turn (us)
        Returns: utility value
        """
        self.nodes_evaluated += 1
        
        # Check terminal state
        is_endgame, p0_score, p1_score = check_endgame(state)
        if is_endgame:
            return self.terminal_value(state, player, opponent, p0_score, p1_score)
        
        # Depth cutoff
        if depth == 0:
            return self.evaluation(state, player, opponent)
        
        moves = get_valid_moves(state, player)
        if not moves:
            # No moves available - opponent's turn
            return self.min_value(state, depth-1, player, opponent, alpha, beta)
        
        # Order moves for better pruning
        moves = self.order_moves(state, moves, player)
        
        value = -float('inf')
        
        for move in moves:
            new_state = deepcopy(state)
            execute_move(new_state, move, player)
            
            value = max(value, self.min_value(new_state, depth-1, player, opponent, alpha, beta))
            
            if value >= beta:
                return value  # β cutoff
            
            alpha = max(alpha, value)
        
        return value
    
    def min_value(self, state, depth, player, opponent, alpha, beta):
        """
        Min player's turn (opponent)
        Returns: utility value
        """
        self.nodes_evaluated += 1
        
        # Check terminal state
        is_endgame, p0_score, p1_score = check_endgame(state)
        if is_endgame:
            return self.terminal_value(state, player, opponent, p0_score, p1_score)
        
        # Depth cutoff
        if depth == 0:
            return self.evaluation(state, player, opponent)
        
        moves = get_valid_moves(state, opponent)
        if not moves:
            # No moves available - our turn
            return self.max_value(state, depth-1, player, opponent, alpha, beta)
        
        # Order moves for opponent (they want to minimize our score)
        moves = self.order_moves_opponent(state, moves, opponent, player)
        
        value = float('inf')
        
        for move in moves:
            new_state = deepcopy(state)
            execute_move(new_state, move, opponent)
            
            value = min(value, self.max_value(new_state, depth-1, player, opponent, alpha, beta))
            
            if value <= alpha:
                return value  # α cutoff
            
            beta = min(beta, value)
        
        return value
    
    def terminal_value(self, state, player, opponent, p0_score, p1_score):
        """
        Evaluate terminal game state
        """
        if player == 1:
            my_score = p0_score
            opp_score = p1_score
        else:
            my_score = p1_score
            opp_score = p0_score
        
        if my_score > opp_score:
            return 1000 + (my_score - opp_score)  # Win
        elif my_score < opp_score:
            return -1000 + (my_score - opp_score)  # Loss
        else:
            return 0  # Tie
    
    def evaluation(self, state, player, opponent):
        """
        Evaluation function for non-terminal states
        """
        # Simple evaluation: piece difference
        if player == 1:
            my_pieces = np.sum(state == 1)
            opp_pieces = np.sum(state == 2)
        else:
            my_pieces = np.sum(state == 2)
            opp_pieces = np.sum(state == 1)
        
        score = (my_pieces - opp_pieces) * 10
        
        # Corner control bonus
        corners = [(0,0), (0,6), (6,0), (6,6)]
        for r, c in corners:
            if state[r, c] == player:
                score += 20
            elif state[r, c] == opponent:
                score -= 20
        
        return score
    
    def order_moves(self, state, moves, player):
        """
        Order moves for Max player (us) - best moves first for better pruning
        """
        from helpers import count_disc_count_change
        
        scored = []
        for move in moves:
            score = 0
            
            # Immediate gain
            gain = count_disc_count_change(state, move, player)
            score += gain * 5
            
            # Corner moves are best
            dest = move.get_dest()
            if dest in [(0,0), (0,6), (6,0), (6,6)]:
                score += 15
            
            # Edge moves are good
            if dest[0] == 0 or dest[0] == 6 or dest[1] == 0 or dest[1] == 6:
                score += 5
            
            scored.append((score, move))
        
        # Sort descending (best moves first)
        scored.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored]
    
    def order_moves_opponent(self, state, moves, opponent, player):
        """
        Order moves for Min player (opponent) - worst moves for us first
        """
        from helpers import count_disc_count_change
        
        scored = []
        for move in moves:
            score = 0
            
            # Opponent's immediate gain (bad for us)
            gain = count_disc_count_change(state, move, opponent)
            score += gain * 5  # Higher = worse for us
            
            # Opponent getting corners is very bad for us
            dest = move.get_dest()
            if dest in [(0,0), (0,6), (6,0), (6,6)]:
                score += 15
            
            scored.append((score, move))
        
        # Sort ascending (worst moves for us first)
        scored.sort(key=lambda x: x[0])
        return [move for _, move in scored]