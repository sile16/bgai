import ctypes
import json
import os

lib_path = os.path.join(os.path.dirname(__file__), "../bgwrapper/libbackgammon.dylib")
bg_lib = ctypes.cdll.LoadLibrary(lib_path)

# Define correct argument and return types for all functions
bg_lib.NewBoard.restype = ctypes.c_uint64

bg_lib.ResetBoard.argtypes = [ctypes.c_uint64]
bg_lib.ResetBoard.restype = None

bg_lib.GetBoardState.argtypes = [ctypes.c_uint64]
bg_lib.GetBoardState.restype = ctypes.c_void_p  # Changed from c_char_p

bg_lib.GetLegalMoves.argtypes = [ctypes.c_uint64]
bg_lib.GetLegalMoves.restype = ctypes.c_void_p  # Changed from c_char_p

bg_lib.ApplyAction.argtypes = [ctypes.c_uint64, ctypes.c_int]
bg_lib.ApplyAction.restype = ctypes.c_uint64

bg_lib.GetEncodedState.argtypes = [ctypes.c_uint64]
bg_lib.GetEncodedState.restype = ctypes.c_void_p  # Changed from c_char_p

bg_lib.FreeString.argtypes = [ctypes.c_void_p]  # Changed from c_char_p
bg_lib.FreeString.restype = None


class BackgammonEnv:
    def __init__(self):
        """Initialize a new backgammon environment."""
        self._board_id = bg_lib.NewBoard()

    def reset(self):
        """Reset the board to its initial state."""
        bg_lib.ResetBoard(self._board_id)

    def get_state(self):
        """Get the current board state as a JSON object."""
        ptr = bg_lib.GetBoardState(self._board_id)
        if ptr:
            try:
                result = ctypes.string_at(ptr).decode("utf-8")
                return json.loads(result)
            finally:
                bg_lib.FreeString(ptr)
        return {}

    def get_legal_moves(self):
        """Get a list of legal moves from the current board state."""
        ptr = bg_lib.GetLegalMoves(self._board_id)
        if ptr:
            try:
                result = ctypes.string_at(ptr).decode("utf-8")
                return json.loads(result)
            finally:
                bg_lib.FreeString(ptr)
        return []

    def step(self, move_idx):
        """Apply a move and return the new board state."""
        new_i, victor, points = bg_lib.ApplyAction(self._board_id, ctypes.c_int(move_idx))
        self._board_id = new_id
        return self.get_state(), victor, points

    def get_encoded_state(self):
        """Get the encoded state representation of the board."""
        ptr = bg_lib.GetEncodedState(self._board_id)
        if ptr:
            try:
                result = ctypes.string_at(ptr).decode("utf-8")
                return json.loads(result)
            finally:
                bg_lib.FreeString(ptr)
        return []
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        # If we had any cleanup for board_id, we'd do it here
        pass