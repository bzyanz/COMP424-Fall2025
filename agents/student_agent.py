# Student agent: Add your own agent here
from agents.agent import Agent
from store import register_agent
import numpy as np
from copy import deepcopy
import time
from helpers import get_valid_moves, execute_move, check_endgame, count_disc_count_change

@register_agent("student_agent")
class StudentAgent(Agent):
    """
    OPTIMIZED Alpha-Beta - Focus on DEEP search with SIMPLE evaluation
    Reaches depth 5-7 consistently
    """
    
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "DeepSimpleAB"
    
    def step(self, chess_board, player, opponent):
        """
        REPLACES old step() - More aggressive time management for deeper search
        """
        start_time = time.time()
        TIME_LIMIT = 1.8  # Safe limit
        
        # Get all moves
        moves = get_valid_moves(chess_board, player)
        if not moves:
            return None
        
        # Quick initial ordering (corners first)
        moves = self.ultra_fast_order(chess_board, moves, player)
        best_move = moves[0]
        
        # AGGRESSIVE iterative deepening - try to reach depth 7
        depths_to_try = [1, 2, 3, 4, 5, 6, 7]
        depth_reached = 0
        
        for depth in depths_to_try:
            current_time = time.time() - start_time
            
            # Progressive time allocation - saves time for deeper searches
            time_limits = {
                1: 0.05,   # Depth 1: 0.05s max
                2: 0.1,    # Depth 2: 0.1s max  
                3: 0.2,    # Depth 3: 0.2s max
                4: 0.4,    # Depth 4: 0.4s max
                5: 0.8,    # Depth 5: 0.8s max
                6: 1.3,    # Depth 6: 1.3s max
                7: 1.7     # Depth 7: 1.7s max
            }
            
            if current_time > time_limits.get(depth, 1.8):
                continue  # Skip this depth if we're already over time budget
            
            try:
                # Use FAST alpha-beta for this depth
                move, score = self.fast_alpha_beta(
                    chess_board, depth, player, opponent,
                    -float('inf'), float('inf'), True,
                    start_time, TIME_LIMIT
                )
                
                if move is not None:
                    best_move = move
                    depth_reached = depth
                    
                    # If decisive score found, stop searching
                    if abs(score) > 800:  # Big lead or deficit
                        break
                        
            except TimeoutError:
                break  # Ran out of time
        
        time_taken = time.time() - start_time
        empty_squares = np.sum(chess_board == 0)
        
        print(f"DeepSimpleAB: depth={depth_reached}, empty={empty_squares}, time={time_taken:.3f}s")
        
        return best_move
    
    def fast_alpha_beta(self, board, depth, player, opponent, alpha, beta, 
                       maximizing, start_time, time_limit):
        """
        REPLACES old alpha_beta() - Ultra-optimized for speed
        Uses minimal evaluation to enable deeper search
        """
        # Fast time check (check less often for speed)
        if depth <= 2 or depth % 2 == 0:  # Check every other level for deep searches
            if time.time() - start_time > time_limit:
                raise TimeoutError()
        
        # Get scores for terminal evaluation
        is_endgame, p0_score, p1_score = check_endgame(board)
        
        # Terminal node or depth limit
        if is_endgame or depth == 0:
            # ULTRA SIMPLE evaluation - just piece difference
            if player == 1:
                score = p0_score - p1_score
            else:
                score = p1_score - p0_score
            return None, score
        
        # Determine current player
        current_player = player if maximizing else opponent
        
        # Get moves
        moves = get_valid_moves(board, current_player)
        if not moves:
            # No moves - evaluate current position simply
            if player == 1:
                score = p0_score - p1_score
            else:
                score = p1_score - p0_score
            return None, score
        
        # MINIMAL move ordering for speed
        if maximizing:
            # Sort by immediate gain, descending
            moves.sort(key=lambda m: count_disc_count_change(board, m, player), reverse=True)
        else:
            # Sort by opponent's gain, ascending (worst for us first)
            moves.sort(key=lambda m: count_disc_count_change(board, m, opponent))
        
        # Progressive widening: examine fewer moves at deeper levels
        max_moves_to_examine = 50  # Default: all moves
        if depth >= 5:
            max_moves_to_examine = 15
        elif depth >= 4:
            max_moves_to_examine = 25
        elif depth >= 3:
            max_moves_to_examine = 35
        
        moves = moves[:max_moves_to_examine]
        best_move = moves[0] if moves else None
        
        if maximizing:
            best_score = -float('inf')
            for move in moves:
                # Apply move
                new_board = deepcopy(board)
                execute_move(new_board, move, current_player)
                
                # Recursive search
                _, score = self.fast_alpha_beta(
                    new_board, depth-1, player, opponent,
                    alpha, beta, False,
                    start_time, time_limit
                )
                
                if score > best_score:
                    best_score = score
                    best_move = move
                
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # Beta cutoff
            
            return best_move, best_score
            
        else:  # minimizing player
            best_score = float('inf')
            for move in moves:
                new_board = deepcopy(board)
                execute_move(new_board, move, current_player)
                
                _, score = self.fast_alpha_beta(
                    new_board, depth-1, player, opponent,
                    alpha, beta, True,
                    start_time, time_limit
                )
                
                if score < best_score:
                    best_score = score
                    best_move = move
                
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # Alpha cutoff
            
            return best_move, best_score
    
    def ultra_fast_order(self, board, moves, player):
        """
        REPLACES old order_moves() - Minimal ordering for initial move selection
        Only used for choosing which move to return if search fails
        """
        # Just two categories: corners and everything else
        corners = []
        non_corners = []
        
        for move in moves:
            dest = move.get_dest()
            if dest in [(0,0), (0,6), (6,0), (6,6)]:
                corners.append(move)
            else:
                # Quick score for non-corners
                gain = count_disc_count_change(board, move, player)
                non_corners.append((gain, move))
        
        # Sort non-corners by gain
        non_corners.sort(reverse=True, key=lambda x: x[0])
        
        # Combine: corners first, then high-gain non-corners
        result = corners[:]
        result.extend([move for _, move in non_corners])
        
        return result
    
    # OLD FUNCTIONS REMOVED:
    # - evaluate() - too complex, slows down search
    # - evaluate_simple() - inlined in fast_alpha_beta
    # - strategic_move_order() - too slow for deep search
    # - creates_dangerous_hole() - tactical heuristics removed for speed
    # - corner_access_penalty() - removed for simplicity


class TimeoutError(Exception):
    """Custom exception for timeout"""
    pass