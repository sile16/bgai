from backgammon_env import BackgammonEnv
from pprint import pprint

env = BackgammonEnv()

env.reset()

state = env.get_state()
print("Initial board state (JSON):", state)

encoded = env.get_encoded_state()
print("Encoded state (2x28 board data):", encoded)

moves = env.get_legal_moves()
print("Legal moves:")

def print_moves(moves):
    for x in moves:
        print(f"Roll: {x['roll']} roll_used: {x['roll_used']} Player: {x['player']}")
        print(f"  White: {x['white_pips']}")
        print(f"    Red: {x['red_pips']}\n")

print_moves(moves)

def player(player_num):
    if player_num == 0:
        return "None"
    elif player_num == 1:
        return "White"
    elif player_num == 2:
        return "Red"



while moves:
    chosen_move = moves[0]['id']  # pick the first move
    next_state, victor, score = env.step(chosen_move)
    if victor:
        print("Game over! Victor:", player(victor), "Score:", score)
        break
    print("Next state after applying move index 0:", next_state)
    moves = env.get_legal_moves()
    print_moves(moves)
else:
    print("No legal moves available.")
