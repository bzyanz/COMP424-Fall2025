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
    Optimized α-β with guaranteed depth 3 when possible, safe fallback when not
    """
    
    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "SmartTimedAlphaBeta"
    
    def step(self, chess_board, player, opponent):
        """
        Strategy: ALWAYS try depth 3 first
        If it's taking too long, interrupt and use depth 2
        Never exceed 1.9 seconds
        """
        start_time = time.time()
        ABSOLUTE_LIMIT = 1.9  # Never exceed this
        DEPTH3_TIMEOUT = 1.5  # If depth 3 takes longer than this, abort
        
        moves = get_valid_moves(chess_board, player)
        if not moves:
            return None
        
        # Quick pre-ordering (fast)
        moves = self.quick_order(chess_board, moves, player)
        best_move = moves[0]  # Default fallback
        
        # TRY DEPTH 3 FIRST (our goal)
        try:
            # Estimate if we have time for depth 3
            empty_squares = np.sum(chess_board == 0)
            estimated_time = self.estimate_time(empty_squares, 3)
            
            if estimated_time < ABSOLUTE_LIMIT * 0.8:  # Safe to try
                #print(f"Attempting depth 3 (estimate: {estimated_time:.2f}s)")
                
                move, value = self.timed_alphabeta(
                    chess_board, 3, player, opponent,
                    -float('inf'), float('inf'), True,
                    start_time, DEPTH3_TIMEOUT
                )
                
                if move is not None:
                    time_taken = time.time() - start_time
                    #print(f"Depth 3 SUCCESS! time={time_taken:.3f}s, value={value}")
                    return move
                else:
                    print("Depth 3 returned None, falling back")
            
        except TimeoutError:
            print("Depth 3 TIMED OUT, falling back to depth 2")
        
        # DEPTH 3 FAILED OR TIMED OUT - TRY DEPTH 2
        current_time = time.time() - start_time
        
        if current_time < 1.7:  # Still have time for depth 2
            try:
                #print("Attempting depth 2")
                
                move, value = self.timed_alphabeta(
                    chess_board, 2, player, opponent,
                    -float('inf'), float('inf'), True,
                    start_time, 1.8
                )
                
                if move is not None:
                    time_taken = time.time() - start_time
                    #print(f"Depth 2 success, time={time_taken:.3f}s")
                    return move
                
            except TimeoutError:
                print("Depth 2 also timed out")
        
        # ULTIMATE FALLBACK: Depth 1 (greedy) - fast and safe
        time_taken = time.time() - start_time
        print(f"Using greedy fallback, time={time_taken:.3f}s")
        return self.greedy_best_move(chess_board, moves, player, opponent)
    
    def timed_alphabeta(self, board, depth, player, opponent, alpha, beta, maximizing, start_time, timeout):
        """α-β with time checking"""
        # Check time first!
        if time.time() - start_time > timeout:
            raise TimeoutError()
        
        # Terminal/depth check
        is_endgame, p0_score, p1_score = check_endgame(board)
        if is_endgame or depth == 0:
            value = self.evaluate(board, player, opponent, p0_score, p1_score)
            return None, value
        
        current_player = player if maximizing else opponent
        moves = get_valid_moves(board, current_player)
        
        if not moves:
            value = self.evaluate(board, player, opponent, *check_endgame(board)[1:])
            return None, value
        
        # Fast ordering
        if maximizing:
            moves = self.quick_order(board, moves, player)
        else:
            moves = self.quick_order(board, moves, opponent)
            if not maximizing:
                moves.reverse()  # For opponent, worst moves for us first
        
        best_move = moves[0]
        
        if maximizing:
            best_value = -float('inf')
            for i, move in enumerate(moves):
                # Periodic time check (every 5 moves for speed)
                if i % 5 == 0 and time.time() - start_time > timeout:
                    raise TimeoutError()
                
                new_board = deepcopy(board)
                execute_move(new_board, move, current_player)
                
                _, value = self.timed_alphabeta(
                    new_board, depth-1, player, opponent,
                    alpha, beta, False,
                    start_time, timeout
                )
                
                if value > best_value:
                    best_value = value
                    best_move = move
                
                alpha = max(alpha, best_value)
                if beta <= alpha:
                    break
            
            return best_move, best_value
            
        else:
            best_value = float('inf')
            for i, move in enumerate(moves):
                if i % 5 == 0 and time.time() - start_time > timeout:
                    raise TimeoutError()
                
                new_board = deepcopy(board)
                execute_move(new_board, move, current_player)
                
                _, value = self.timed_alphabeta(
                    new_board, depth-1, player, opponent,
                    alpha, beta, True,
                    start_time, timeout
                )
                
                if value < best_value:
                    best_value = value
                    best_move = move
                
                beta = min(beta, best_value)
                if beta <= alpha:
                    break
            
            return best_move, best_value
    
    def evaluate(self, board, player, opponent, p0_score=None, p1_score=None):
        """Fast evaluation"""
        if p0_score is None or p1_score is None:
            _, p0_score, p1_score = check_endgame(board)
        
        if player == 1:
            my_score = p0_score
            opp_score = p1_score
        else:
            my_score = p1_score
            opp_score = p0_score
        
        # Terminal state bonus
        if my_score > opp_score:
            return 1000 + (my_score - opp_score)
        elif my_score < opp_score:
            return -1000 + (my_score - opp_score)
        
        # Non-terminal: piece difference + corners
        score = (my_score - opp_score) * 10
        
        corners = [(0,0), (0,6), (6,0), (6,6)]
        for r, c in corners:
            if board[r, c] == player:
                score += 15
            elif board[r, c] == opponent:
                score -= 15
        
        return score
    
    def quick_order(self, board, moves, player):
        """Very fast move ordering"""
        scored = []
        for move in moves:
            score = count_disc_count_change(board, move, player) * 3
            
            dest = move.get_dest()
            if dest in [(0,0), (0,6), (6,0), (6,6)]:
                score += 20
            
            scored.append((score, move))
        
        scored.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in scored]
    
    def greedy_best_move(self, board, moves, player, opponent):
        """Fast greedy fallback (depth 1)"""
        best_move = moves[0]
        best_score = -float('inf')
        
        for move in moves:
            new_board = deepcopy(board)
            execute_move(new_board, move, player)
            
            score = self.evaluate(new_board, player, opponent)
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move
    
    def estimate_time(self, empty_squares, depth):
        """
        Estimate time needed based on board state
        Less empty squares = faster search
        """
        # Empirical estimates based on your 1.296s for depth 3
        base_time = 1.3  # Your depth 3 took 1.296s on some board
        
        # Adjust for empty squares (more empties = more moves = slower)
        if empty_squares > 40:  # Early game
            return base_time * 1.5
        elif empty_squares > 20:  # Mid game
            return base_time
        else:  # End game
            return base_time * 0.7


class TimeoutError(Exception):
    pass