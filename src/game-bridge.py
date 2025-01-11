import ctypes
import numpy as np
import json
from pathlib import Path
from typing import Tuple, List, Optional

class BackgammonEnv:
    """Bridge between Python and Go backgammon implementation."""
    
    def __init__(self):
        # Load the shared library
        lib_path = Path(__file__).parent / "lib" / "libbackgammon.dylib"
        if not lib_path.exists():
            lib_path = Path(__file__).parent / "lib" / "libbackgammon.so"
        
        self.lib = ctypes.CDLL(str(lib_path))
        
        # Define function signatures
        self.lib.NewGame.restype = ctypes.c_uint64
        self.lib.NewGame.argtypes = []
        
        self.lib.GetEncodedState.restype = ctypes.c_void_p
        self.lib.GetEncodedState.argtypes = [ctypes.c_uint64]
        
        self.lib.GetLegalMoves.restype = ctypes.c_void_p
        self.lib.GetLegalMoves.argtypes = [ctypes.c_uint64]
        
        self.lib.ApplyMove.restype = ctypes.c_uint64
        self.lib.ApplyMove.argtypes = [ctypes.c_uint64, ctypes.c_int]
        
        self.lib.IsGameOver.restype = ctypes.c_bool
        self.lib.IsGameOver.argtypes = [ctypes.c_uint64]
        
        self.lib.GetGameResult.restype = ctypes.c_float
        self.lib.GetGameResult.argtypes = [ctypes.c_uint64]
        
        self.lib.FreeString.argtypes = [ctypes.c_void_p]
        self.lib.FreeString.restype = None
        
        # Initialize game
        self.game_id = self.lib.NewGame()
    
    def get_state(self) -> np.ndarray:
        """Get the encoded state from Go."""
        ptr = self.lib.GetEncodedState(self.game_id)
        try:
            result = ctypes.string_at(ptr)
            state = np.array(json.loads(result), dtype=np.float32)
            return state
        finally:
            self.lib.FreeString(ptr)
    
    def get_legal_moves(self) -> List[dict]:
        """Get legal moves and their information from Go."""
        ptr = self.lib.GetLegalMoves(self.game_id)
        try:
            result = ctypes.string_at(ptr)
            moves = json.loads(result)
            return moves
        finally:
            self.lib.FreeString(ptr)
    
    def apply_move(self, move_idx: int) -> np.ndarray:
        """Apply a move and get the resulting state."""
        self.game_id = self.lib.ApplyMove(self.game_id, move_idx)
        return self.get_state()
    
    def is_terminal(self) -> bool:
        """Check if the game is over."""
        return bool(self.lib.IsGameOver(self.game_id))
    
    def get_result(self) -> float:
        """Get the game result (1.0 for white win, -1.0 for red win)."""
        return float(self.lib.GetGameResult(self.game_id))
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'game_id'):
            self.lib.FreeGame(self.game_id)

class BackgammonGame:
    """High-level game interface for AI training."""
    
    def __init__(self):
        self.env = BackgammonEnv()
        
    def get_initial_state(self) -> np.ndarray:
        """Get the initial game state."""
        return self.env.get_state()
    
    def get_next_state(self, move_idx: int) -> Tuple[np.ndarray, Optional[float]]:
        """Apply move and return (new_state, reward if terminal, else None)."""
        state = self.env.apply_move(move_idx)
        if self.env.is_terminal():
            return state, self.env.get_result()
        return state, None
    
    def get_valid_moves(self) -> Tuple[List[dict], np.ndarray]:
        """Get legal moves and their mask."""
        moves = self.env.get_legal_moves()
        mask = np.zeros(256, dtype=np.float32)  # Maximum possible moves
        for move in moves:
            mask[move['id']] = 1.0
        return moves, mask
