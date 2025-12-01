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
    Alpha-Beta Pruning with Time Management
    """
    
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "TimedAlphaBeta"
    
    def step(self, chess_board, player, opponent):
        """
        Iterative deepening with strict time limit
        """
        start_time = time.time()
        TIME_LIMIT = 1.8  # Conservative limit (2s max, leave 0.2s buffer)
        
        # Get all valid moves
        moves = get_valid_moves(chess_board, player)
        if not moves:
            return None
        
        # Order moves by immediate gain for better early decisions
        moves = self.order_moves(chess_board, moves, player)
        
        # Start with best immediate move as fallback
        best_move = moves[0]
        best_score = -float('inf')
        
        # Iterative deepening: start shallow, go deeper if time allows
        depth = 1
        max_reached_depth = 0
        
        while True:
            current_time = time.time() - start_time
            
            # Check if we have time for another depth level
            # Use more conservative checking as depth increases
            time_per_depth = 0.1 * (2 ** depth)  # Exponential time estimate
            if current_time + time_per_depth > TIME_LIMIT:
                break
            
            try:
                # Run alpha-beta at current depth with timeout checking
                move, score = self.alpha_beta(
                    chess_board, depth, player, opponent,
                    -float('inf'), float('inf'), True,
                    start_time, TIME_LIMIT
                )
                
                if move is not None:
                    best_move = move
                    best_score = score
                    max_reached_depth = depth
                    
                    # If we found a clearly winning move, use it immediately
                    if score > 500:  # Arbitrary large winning score
                        break
                
                depth += 1
                
                # Safety: don't go too deep even if time allows
                if depth > 8:
                    break
                    
            except TimeoutError:
                # Time ran out during search, use best found so far
                break
        
        time_taken = time.time() - start_time
        print(f"TimedAlphaBeta: depth={max_reached_depth}, time={time_taken:.3f}s, score={best_score:.1f}")
        
        return best_move
    
    def alpha_beta(self, board, depth, player, opponent, alpha, beta, maximizing, 
                   start_time, time_limit):
        """
        Alpha-Beta with time checking
        """
        # Check time first!
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        
        # Check terminal state
        is_endgame, p0_score, p1_score = check_endgame(board)
        if is_endgame or depth == 0:
            score = self.evaluate(board, player, opponent, p0_score, p1_score)
            return None, score
        
        # Determine current player
        current_player = player if maximizing else opponent
        
        # Get moves for current player
        moves = get_valid_moves(board, current_player)
        if not moves:
            # If no moves, skip turn
            score = self.evaluate(board, player, opponent, *check_endgame(board)[1:])
            return None, score
        
        # Order moves
        if maximizing:
            moves = self.order_moves(board, moves, player)
        else:
            moves = self.order_moves(board, moves, opponent)
            moves.reverse()  # For minimizing, want worst for opponent first
        
        best_move = moves[0]
        
        if maximizing:
            best_score = -float('inf')
            for move in moves:
                # Quick time check
                if time.time() - start_time > time_limit:
                    raise TimeoutError()
                
                # Apply move
                new_board = deepcopy(board)
                execute_move(new_board, move, current_player)
                
                # Recursive search
                _, score = self.alpha_beta(
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
                if time.time() - start_time > time_limit:
                    raise TimeoutError()
                
                new_board = deepcopy(board)
                execute_move(new_board, move, current_player)
                
                _, score = self.alpha_beta(
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
    
    def evaluate(self, board, player, opponent, p0_score=None, p1_score=None):
        """
        Evaluation function with terminal state handling
        """
        if p0_score is None or p1_score is None:
            _, p0_score, p1_score = check_endgame(board)
        
        # Terminal state evaluation
        if player == 1:
            my_score = p0_score
            opp_score = p1_score
        else:
            my_score = p1_score
            opp_score = p0_score
        
        if my_score > opp_score:
            return 1000 + (my_score - opp_score)  # Winning
        elif my_score < opp_score:
            return -1000 - (opp_score - my_score)  # Losing
        
        # Non-terminal state evaluation
        player_count = np.sum(board == player)
        opponent_count = np.sum(board == opponent)
        score = (player_count - opponent_count) * 10
        
        # Corner control (very valuable)
        corners = [(0,0), (0,6), (6,0), (6,6)]
        for r, c in corners:
            if board[r, c] == player:
                score += 25  # Increased importance
            elif board[r, c] == opponent:
                score -= 25
        
        # Edge control
        for i in range(7):
            if board[0, i] == player: score += 3
            if board[6, i] == player: score += 3
            if board[i, 0] == player: score += 3
            if board[i, 6] == player: score += 3
            
            if board[0, i] == opponent: score -= 3
            if board[6, i] == opponent: score -= 3
            if board[i, 0] == opponent: score -= 3
            if board[i, 6] == opponent: score -= 3
        
        # Mobility
        player_moves = len(get_valid_moves(board, player))
        opponent_moves = len(get_valid_moves(board, opponent))
        score += (player_moves - opponent_moves) * 3
        
        # Center control (minor factor)
        center = [(3,3), (3,2), (2,3), (2,2)]
        for r, c in center:
            if board[r, c] == player: score += 1
            elif board[r, c] == opponent: score -= 1
        
        return score
    
    def order_moves(self, board, moves, player):
        """
        Order moves by immediate gain and strategic value
        """
        scored = []
        for move in moves:
            score = 0
            
            # Immediate disc gain
            gain = count_disc_count_change(board, move, player)
            score += gain * 5
            
            # Destination bonuses
            dest = move.get_dest()
            
            # Corner moves are best
            if dest in [(0,0), (0,6), (6,0), (6,6)]:
                score += 20
            
            # Edge moves are good
            elif dest[0] == 0 or dest[0] == 6 or dest[1] == 0 or dest[1] == 6:
                score += 8
            
            # Avoid dangerous squares next to opponent corners
            opponent = 2 if player == 1 else 1
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                r, c = dest[0] + dr, dest[1] + dc
                if 0 <= r < 7 and 0 <= c < 7:
                    if (r, c) in [(0,0), (0,6), (6,0), (6,6)] and board[r, c] == opponent:
                        score -= 15  # Big penalty for giving opponent corner access
            
            scored.append((score, move))
        
        # Sort by score descending
        scored.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored]


class TimeoutError(Exception):
    """Custom exception for timeout"""
    pass