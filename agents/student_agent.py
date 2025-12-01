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
    Simplified Alpha-Beta based on TkAtaxx insights
    Simple evaluation (piece difference only) + strategic move ordering
    """
    
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "TkAtaxxStyle"
    
    def step(self, chess_board, player, opponent):
        """
        Simple iterative deepening with TkAtaxx-style evaluation
        """
        start_time = time.time()
        TIME_LIMIT = 1.8
        
        moves = get_valid_moves(chess_board, player)
        if not moves:
            return None
        
        # Strategic move ordering based on game insights
        moves = self.strategic_move_order(chess_board, moves, player, opponent)
        
        best_move = moves[0]
        best_score = -float('inf')
        
        # Iterative deepening
        depth = 1
        max_depth = 0
        
        while True:
            current_time = time.time() - start_time
            if current_time > TIME_LIMIT * 0.7:
                break
            
            # Simple time estimation - depth 1-2 are fast, 3+ slower
            if depth == 1 and current_time > 0.1:
                pass
            elif depth == 2 and current_time > 0.3:
                break
            elif depth == 3 and current_time > 0.8:
                break
            elif depth == 4 and current_time > 1.5:
                break
            
            try:
                move, score = self.alpha_beta(
                    chess_board, depth, player, opponent,
                    -float('inf'), float('inf'), True,
                    start_time, TIME_LIMIT
                )
                
                if move is not None:
                    best_move = move
                    best_score = score
                    max_depth = depth
                    
                    # If clearly winning, stop
                    if score > 900:
                        break
                
                depth += 1
                if depth > 6:  # Reasonable max
                    break
                    
            except TimeoutError:
                break
        
        time_taken = time.time() - start_time
        print(f"TkAtaxxStyle: depth={max_depth}, time={time_taken:.3f}s")
        
        return best_move
    
    def alpha_beta(self, board, depth, player, opponent, alpha, beta, 
                   maximizing, start_time, time_limit):
        """Alpha-beta with simple evaluation"""
        # Time check
        if time.time() - start_time > time_limit:
            raise TimeoutError()
        
        # Terminal check
        is_endgame, p0_score, p1_score = check_endgame(board)
        if is_endgame or depth == 0:
            score = self.evaluate_simple(board, player, opponent, p0_score, p1_score)
            return None, score
        
        current_player = player if maximizing else opponent
        moves = get_valid_moves(board, current_player)
        
        if not moves:
            # No moves - evaluate current position
            score = self.evaluate_simple(board, player, opponent, *check_endgame(board)[1:])
            return None, score
        
        # Move ordering: strategic for maximizing, reverse for minimizing
        if maximizing:
            moves = self.strategic_move_order(board, moves, player, opponent)
        else:
            moves = self.strategic_move_order(board, moves, opponent, player)
            moves.reverse()  # Worst moves first for opponent
        
        best_move = moves[0]
        
        if maximizing:
            best_score = -float('inf')
            for move in moves:
                # Quick time check
                if time.time() - start_time > time_limit:
                    raise TimeoutError()
                
                new_board = deepcopy(board)
                execute_move(new_board, move, current_player)
                
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
                    break
            
            return best_move, best_score
            
        else:  # minimizing
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
                    break
            
            return best_move, best_score
    
    def evaluate_simple(self, board, player, opponent, p0_score=None, p1_score=None):
        """
        TkAtaxx-style evaluation: piece difference only
        """
        if p0_score is None or p1_score is None:
            _, p0_score, p1_score = check_endgame(board)
        
        if player == 1:
            my_score = p0_score
            opp_score = p1_score
        else:
            my_score = p1_score
            opp_score = p0_score
        
        # Terminal state evaluation
        if my_score > opp_score:
            return 1000 + (my_score - opp_score)
        elif my_score < opp_score:
            return -1000 - (opp_score - my_score)
        
        # Non-terminal: just piece difference (like TkAtaxx)
        return my_score - opp_score
    
    def strategic_move_order(self, board, moves, player, opponent):
        """
        Strategic move ordering based on game insights
        """
        scored = []
        for move in moves:
            score = 0
            
            # 1. Immediate gain (most important)
            gain = count_disc_count_change(board, move, player)
            score += gain * 10
            
            # 2. Corner control (from strategy text)
            dest = move.get_dest()
            if dest in [(0,0), (0,6), (6,0), (6,6)]:
                score += 30  # Corners are extremely valuable
            
            # 3. Avoid creating "holes of one"
            src = move.get_src()
            dest_r, dest_c = dest
            
            # Check if move creates a dangerous single-square hole
            if self.creates_dangerous_hole(board, src, dest, player, opponent):
                score -= 25
            
            # 4. Edge control (secondary)
            if dest_r == 0 or dest_r == 6 or dest_c == 0 or dest_c == 6:
                score += 5
            
            # 5. Don't give opponent corner access
            opp_corner_penalty = self.corner_access_penalty(board, dest, opponent)
            score -= opp_corner_penalty
            
            scored.append((score, move))
        
        # Sort by score descending
        scored.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored]
    
    def creates_dangerous_hole(self, board, src, dest, player, opponent):
        """
        Check if move creates a dangerous single-square hole
        Based on "Hole of one, yes, hole of two, no!"
        """
        src_r, src_c = src
        dest_r, dest_c = dest
        
        # If it's a jump (movement), check if source becomes a dangerous hole
        if abs(dest_r - src_r) == 2 or abs(dest_c - src_c) == 2:
            # Check if source square becomes isolated
            isolated = True
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = src_r + dr, src_c + dc
                    if 0 <= nr < 7 and 0 <= nc < 7:
                        if board[nr, nc] == player:
                            isolated = False
                            break
                if not isolated:
                    break
            
            if isolated:
                # Check if opponent can jump into this hole
                opp_moves = get_valid_moves(board, opponent)
                for opp_move in opp_moves:
                    opp_dest = opp_move.get_dest()
                    if opp_dest == (src_r, src_c):
                        return True  # Dangerous hole!
        
        return False
    
    def corner_access_penalty(self, board, dest, opponent):
        """
        Penalize moves that give opponent access to corners
        """
        penalty = 0
        dest_r, dest_c = dest
        
        # Check adjacent squares
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = dest_r + dr, dest_c + dc
                if 0 <= nr < 7 and 0 <= nc < 7:
                    # If adjacent to opponent's corner, big penalty
                    if (nr, nc) in [(0,0), (0,6), (6,0), (6,6)]:
                        if board[nr, nc] == opponent:
                            penalty += 20
        
        return penalty


class TimeoutError(Exception):
    pass