from backgammon_env import BackgammonEnv

env = BackgammonEnv()

env.hello_world()

env.reset()

state = env.get_state()
print("Initial board state (JSON):", state)

encoded = env.get_encoded_state()
print("Encoded state (2x28 board data):", encoded)

moves = env.get_legal_moves()
print("Legal moves:", moves)

if moves:
    chosen_move = moves[0]['id']  # pick the first move
    next_state = env.step(chosen_move)
    print("Next state after applying move index 0:", next_state)
else:
    print("No legal moves available.")
