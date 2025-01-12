import ctypes
import json
import os
from typing import Tuple, Dict, List, Any, Optional 
import numpy as np
import logging

class BackgammonStateError(Exception):
    """Custom exception for state encoding/decoding errors."""
    pass


class ActionResult(ctypes.Structure):
    _fields_ = [
        ("new_id", ctypes.c_ulonglong),
        ("victor", ctypes.c_int),
        ("points", ctypes.c_int)
    ]

class MoveInfo(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("player", ctypes.c_float),
        ("roll", ctypes.c_float * 4),
        ("roll_used", ctypes.c_float * 4),
        ("white_pips", ctypes.c_float * 26),
        ("red_pips", ctypes.c_float * 26)
    ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the C struct to a Python dictionary."""
        return {
            "id": self.id,
            "player": self.player,
            "roll": list(self.roll),
            "roll_used": list(self.roll_used),
            "white_pips": list(self.white_pips),
            "red_pips": list(self.red_pips)
        }

class MoveList(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_int),
        ("moves", ctypes.POINTER(MoveInfo))
    ]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.moves:
            bg_lib.FreeMoveList(self)
    
    def to_list(self) -> List[Dict[str, Any]]:
        """Convert the C array of moves to a Python list of dictionaries."""
        if self.count <= 0:
            return []
        return [self.moves[i].to_dict() for i in range(self.count)]

lib_path = os.path.join(os.path.dirname(__file__), "../bgwrapper/libbackgammon.dylib")
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"Could not find library at {lib_path}")
try:
    bg_lib = ctypes.cdll.LoadLibrary(lib_path)
except Exception as e:
    print(f"Failed to load library: {e}")
    raise

# Define correct argument and return types for all functions
bg_lib.NewBoard.restype = ctypes.c_uint64

bg_lib.ResetBoard.argtypes = [ctypes.c_uint64]
bg_lib.ResetBoard.restype = None

bg_lib.GetBoardState.argtypes = [ctypes.c_uint64]
bg_lib.GetBoardState.restype = ctypes.c_void_p  # Changed from c_char_p

bg_lib.GetLegalMoves.argtypes = [ctypes.c_uint64]
bg_lib.GetLegalMoves.restype = MoveList

bg_lib.FreeMoveList.argtypes = [MoveList]
bg_lib.FreeMoveList.restype = None

bg_lib.GetEncodedState.argtypes = [ctypes.c_uint64]
bg_lib.GetEncodedState.restype = ctypes.c_void_p  # Changed from c_char_p

bg_lib.FreeString.argtypes = [ctypes.c_void_p]  # Changed from c_char_p
bg_lib.FreeString.restype = None

bg_lib.ApplyAction.restype = ActionResult
bg_lib.ApplyAIAction.restype = ActionResult

# Add to function definitions at top:
bg_lib.SetRandomSeed.argtypes = [ctypes.c_uint64]
bg_lib.SetRandomSeed.restype = None

# Add new function definitions for AI choosers
bg_lib.SetWhiteAIChooser.argtypes = [ctypes.c_int]
bg_lib.SetWhiteAIChooser.restype = None

bg_lib.SetRedAIChooser.argtypes = [ctypes.c_int]
bg_lib.SetRedAIChooser.restype = None

# Add GetEncodedStateForNN definition
bg_lib.GetEncodedStateForNN.argtypes = [ctypes.c_uint64]
bg_lib.GetEncodedStateForNN.restype = ctypes.c_void_p  # Same as other state getters


class BackgammonEnv:
     # AI type constants
    SMART_AI = 0
    RANDOM_AI = 1
    WORST_AI = 2
    EXPECTED_CHANNELS = 30
    BOARD_SIZE = 24

    def __init__(self, seed: int = None):
        """Initialize a new backgammon environment."""
        self._board_id = bg_lib.NewBoard()
        self._legal_moves_cache = None

        # by default seed is alreeady random by deafult in go 1.20+
        # this lets us set a seed for reproducibility
        if seed is not None:
            bg_lib.SetRandomSeed(ctypes.c_uint64(seed))
        self._legal_moves_cache = None

    def get_state_for_nn(self) -> Dict[str, Any]:
        """Get encoded state representation specifically formatted for neural network input.
        
        Returns:
            Dict containing:
                state_data: np.ndarray of shape (30, 24) containing encoded board state
                legal_moves: List[int] of legal move indices
                is_terminal: bool indicating if state is terminal
                value: float of current value if terminal
                game_phase: int indicating game phase
                
        Raises:
            BackgammonStateError: If state encoding fails or validation fails
        """
        try:
            if not self._board_id:
                raise ValueError("Invalid board_id")
            
            board_id = ctypes.c_uint64(self._board_id)
            ptr = bg_lib.GetEncodedStateForNN(board_id)
            if not ptr:
                raise BackgammonStateError("Failed to get encoded state from C library")
            
            result = self._get_string_result(ptr)
            if not result:
                raise BackgammonStateError("Empty result from C library")
            
            state = json.loads(result)
            
            # Validate state format
            self._validate_state_format(state)
            
            # Convert state_data to numpy array with proper shape
            state["state_data"] = np.array(state["state_data"], dtype=np.float32).reshape(
                self.EXPECTED_CHANNELS, self.BOARD_SIZE
            )
            
            return state
            
        except json.JSONDecodeError as e:
            raise BackgammonStateError(f"Failed to decode state JSON: {e}")
        except Exception as e:
            raise BackgammonStateError(f"Unexpected error getting NN state: {e}")
    
    def _validate_state_format(self, state: Dict[str, Any]) -> None:
        """Validate the format of the encoded state.
        
        Args:
            state: Dict containing encoded state from C library
            
        Raises:
            BackgammonStateError: If state format is invalid
        """
        required_keys = {"state_data", "shape", "legal_moves", "is_terminal", "value", "game_phase"}
        missing_keys = required_keys - set(state.keys())
        if missing_keys:
            raise BackgammonStateError(f"Missing required keys in state: {missing_keys}")
            
        # Validate state_data
        if not isinstance(state["state_data"], list):
            raise BackgammonStateError("state_data must be a list")
            
        expected_length = self.EXPECTED_CHANNELS * self.BOARD_SIZE
        if len(state["state_data"]) != expected_length:
            raise BackgammonStateError(
                f"state_data has wrong length. Expected {expected_length}, got {len(state['state_data'])}"
            )
            
        # Validate shape
        if not isinstance(state["shape"], list) or state["shape"] != [self.EXPECTED_CHANNELS, self.BOARD_SIZE]:
            raise BackgammonStateError(f"Invalid shape. Expected [{self.EXPECTED_CHANNELS}, {self.BOARD_SIZE}]")
            
        # Validate legal_moves
        if not isinstance(state["legal_moves"], list):
            raise BackgammonStateError("legal_moves must be a list")
            
        # Validate numeric fields
        try:
            for move in state["legal_moves"]:
                if not isinstance(move, (int, float)) or move < 0:
                    raise BackgammonStateError(f"Invalid legal move value: {move}")
                    
            if not isinstance(state["value"], (int, float)):
                raise BackgammonStateError(f"Invalid value type: {type(state['value'])}")
                
            if not isinstance(state["game_phase"], int) or state["game_phase"] not in {0, 1, 2, 3}:
                raise BackgammonStateError(f"Invalid game phase: {state['game_phase']}")
                
        except TypeError as e:
            raise BackgammonStateError(f"Type validation failed: {e}")
            
    def get_observation(self) -> np.ndarray:
        """Get current observation as numpy array for neural network.
        
        Returns:
            np.ndarray: Shape (30, 24) containing encoded board state
            
        Raises:
            BackgammonStateError: If observation creation fails
        """
        try:
            state = self.get_state_for_nn()
            return state["state_data"]
        except BackgammonStateError as e:
            raise BackgammonStateError(f"Failed to get observation: {e}")
    
    def get_legal_actions(self) -> List[int]:
        """Get list of legal action indices.
        
        Returns:
            List[int]: Legal move indices
            
        Raises:
            BackgammonStateError: If getting legal moves fails
        """
        try:
            state = self.get_state_for_nn()
            return state["legal_moves"]
        except BackgammonStateError as e:
            raise BackgammonStateError(f"Failed to get legal actions: {e}")
    
    def is_terminal(self) -> Tuple[bool, float]:
        """Check if state is terminal and return value if it is.
        
        Returns:
            Tuple[bool, float]: (is_terminal, value)
            
        Raises:
            BackgammonStateError: If terminal state check fails
        """
        try:
            state = self.get_state_for_nn()
            return state["is_terminal"], state["value"]
        except BackgammonStateError as e:
            raise BackgammonStateError(f"Failed to check terminal state: {e}")
    
    def step_with_action(self, action_index: int) -> Tuple[Dict[str, Any], float, bool]:
        """Take a step using encoded action index.
        
        Args:
            action_index: Integer index of the action to take
            
        Returns:
            state: New state representation
            reward: Reward for the action
            done: Whether episode is complete
        """
        state, victor, points = self.step(action_index)
        
        reward = 0.0
        if victor:
            reward = points if victor == 1 else -points
            
        done = victor is not None
        return self.get_state_for_nn(), reward, done
    
    def get_observation(self) -> np.ndarray:
        """Get current observation as numpy array for neural network."""
        state = self.get_state_for_nn()
        return np.array(state["state_data"]).reshape(state["shape"])
    
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """Get shape of observation tensor."""
        return (30, 24)  # Channels x Board positions
    
    @property
    def action_space_size(self) -> int:
        """Get size of action space."""
        return 7128  # Total number of possible moves (can be calculated)

    def reset(self):
        """Reset the board to its initial state."""
        bg_lib.ResetBoard(self._board_id)
        self._legal_moves_cache = None

    def set_white_ai(self, ai_type: int):
        """Set the AI type for the White player.
        
        Args:
            ai_type: One of SMART_AI, RANDOM_AI, or WORST_AI
        """
        bg_lib.SetWhiteAIChooser(ctypes.c_int(ai_type))

    def set_red_ai(self, ai_type: int):
        """Set the AI type for the Red player.
        
        Args:
            ai_type: One of SMART_AI, RANDOM_AI, or WORST_AI
        """
        bg_lib.SetRedAIChooser(ctypes.c_int(ai_type))

    def _get_string_result(self, ptr: int) -> str:
        """Helper to handle string results from C."""
        if ptr:
            try:
                return ctypes.string_at(ptr).decode("utf-8")
            finally:
                bg_lib.FreeString(ptr)
        return ""

    def get_state(self) -> Dict[str, Any]:
        """Get the current board state."""
        ptr = bg_lib.GetBoardState(self._board_id)
        result = self._get_string_result(ptr)
        return json.loads(result) if result else {}

    def get_legal_moves(self) -> List[Dict[str, Any]]:
        """Get list of legal moves from current state."""
        if self._legal_moves_cache is None:
            with bg_lib.GetLegalMoves(self._board_id) as move_list:
                self._legal_moves_cache = move_list.to_list()
        return self._legal_moves_cache

    def step(self, move_idx: int) -> Tuple[Dict[str, Any], int, int]:
        """Apply a move and return (new_state, victor, points)."""
        result = bg_lib.ApplyAction(self._board_id, ctypes.c_int(move_idx))
        self._board_id = result.new_id
        self._legal_moves_cache = None
        return self.get_encoded_state(), result.victor, result.points

    def step_ai(self) -> Tuple[Dict[str, Any], int, int]:
        result = bg_lib.ApplyAIAction(self._board_id)
        self._board_id = result.new_id
        self._legal_moves_cache = None
        return self.get_encoded_state(), result.victor, result.points


    def get_encoded_state(self) -> Dict[str, Any]:
        """Get encoded state representation."""
        ptr = bg_lib.GetEncodedState(self._board_id)
        result = self._get_string_result(ptr)
        return json.loads(result) if result else {}
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        # If we had any cleanup for board_id, we'd do it here
        pass