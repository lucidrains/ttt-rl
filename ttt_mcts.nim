
# following the pull request here https://github.com/antirez/ttt-rl/pull/2
# from @robitec97

import std / [
  os,
  math,
  sequtils,
  strutils,
  strformat,
  random,
  terminal
]

randomize()

# Neural network parameters.

const
  NN_INPUT_SIZE = 18
  NN_HIDDEN_SIZE = 100
  NN_OUTPUT_SIZE = 9
  LEARNING_RATE = 0.1

# Game board representation.

type
  GameState = object
    board: array[9, char]
    current_player: int

  GameStateRef = ref GameState

# Initialize game state with an empty board.

proc init_game(state: GameStateRef) =
  for i in 0..<9:
    state.board[i] = '.'

  state.current_player = 0 # Player (X) goes first

# Show board on screen in ASCII "art"...

proc display_board(state: GameStateRef) =
  for row in 0..<3:
    echo &"{state.board[row*3]}{state.board[row*3 + 1]}{state.board[row*3 + 2]} {(row*3)}{(row*3 + 1)}{(row*3 + 2)}"

  echo "\n"

# Count valid moves

proc count_valid_moves(state: GameStateRef): int =
  for tile in state.board:
    if tile == '.':
      result.inc

# Current player symbol

proc current_player_symbol(state: GameStateRef): char =
  if state.current_player == 0: 'X' else: 'O'

proc next_player(state: GameStateRef): int =
  state.current_player xor 1

proc advance_player(state: GameStateRef) =
  state.current_player = state.next_player

# Convert board state to neural network inputs. Note that we use
# a peculiar encoding I descrived here:
# https://www.youtube.com/watch?v=EXbgUXt8fFU
#
# Instead of one-hot encoding, we can represent N different categories
# as different bit patterns. In this specific case it's trivial:
#
# 00 = empty
# 10 = X
# 01 = O
#
# Two inputs per symbol instead of 3 in this case, but in the general case
# this reduces the input dimensionality A LOT.
#
# LEARNING OPPORTUNITY: You may want to learn (if not already aware) of
# different ways to represent non scalar inputs in neural networks:
# One hot encoding, learned embeddings, and even if it's just my random
# exeriment this "permutation coding" that I'm using here.

proc board_to_inputs(state: GameStateRef, inputs: var array[NN_INPUT_SIZE, float]) =

  for i in 0..<9:
    (inputs[i*2], inputs[i*2+1]) = if state.board[i] == '.':
      (0.0, 0.0)
    elif state.board[i] == 'X':
      (1.0, 0.0)
    else:
      (0.0, 1.0)

# Check if the game is over (win or tie).
# Very brutal but fast enough.

proc check_game_over(state: GameStateRef, winner: var char): bool =
  for i in 0..<3:
    if (
      state.board[i*3] != '.' and
      state.board[i*3] == state.board[i*3+1] and
      state.board[i*3+1] == state.board[i*3+2]
    ):
      winner = state.board[i*3]
      return true

  # Check columns

  for i in 0..<3:
    if (
      state.board[i] != '.' and
      state.board[i] == state.board[i+3] and
      state.board[i+3] == state.board[i+6]
    ):
      winner = state.board[i]
      return true

  # Check diagonals

  if (
    (state.board[0] != '.') and
    (state.board[0] == state.board[4]) and
    (state.board[4] == state.board[8])
  ):
    winner = state.board[0]
    return true

  if (
    (state.board[2] != '.') and
    (state.board[2] == state.board[4]) and
    (state.board[4] == state.board[6])
  ):
    winner = state.board[2]
    return true

  # Check for tie (no free tiles left).

  var empty_tiles = 0
  for tile in state.board:
    if tile == '.':
      empty_tiles.inc

  if empty_tiles == 0:
    winner = 'T'
    return true

  return false # Game Continues

# Get a random valid move, this is used for training
# against a random opponent. Note: this function will loop forever
# if the board is full, but here we want simple code.

proc get_random_move(state: GameStateRef): int =
  while true:
    let move = rand(8)

    if (state.board[move] != '.'):
      continue

    return move

# Neural network structure. For simplicity we have just
# one hidden layer and fixed sizes (see defines above).
# However for this problem going deeper than one hidden layer
# is useless.

type
  NeuralNetwork = object
    # Weights and Biases
    weights_ih: array[NN_INPUT_SIZE * NN_HIDDEN_SIZE, float]
    weights_ho: array[NN_HIDDEN_SIZE * NN_OUTPUT_SIZE, float]
    biases_h: array[NN_HIDDEN_SIZE, float]
    biases_o: array[NN_OUTPUT_SIZE, float]

    # Activations are part of the structure itself for simplicity.
    inputs: array[NN_INPUT_SIZE, float]
    hiddens: array[NN_HIDDEN_SIZE, float]
    raw_logits: array[NN_OUTPUT_SIZE, float] # Outputs before softmax().
    outputs: array[NN_OUTPUT_SIZE, float] # Outputs after softmax().

  NeuralNetworkRef = ref NeuralNetwork

# Monte Carlo Tree Search related

const
  MCTS_SIMULATIONS = 1000
  UCB_CONSTANT = 1.414  # Exploration parameter, usnig the most common used value
  LARGE_MAX_VALUE = 1e9

type
  MCTSNode = object
    state: GameStateRef    # Game state at this node
    visits: int            # Number of times this node has been visited
    score: float           # Total score from simulations
    move: int              # Move that led to this state (-1 for root)
    num_children: int      # Number of child nodes
    children: seq[MCTSNodeRef] # Seq of child nodes
    expanded: bool         # Flag to indicate if node is expanded

  MCTSNodeRef = ref MCTSNode

proc create_mcts_node(state: GameStateRef, move: int): MCTSNodeRef =
  MCTSNodeRef(state: state, move: move)

# Expand a node by creating all possible child nodes

proc expand(node: MCTSNodeRef) =
  if node.expanded:
    return

  let valid_moves = node.state.count_valid_moves()

  if valid_moves == 0: # if no valid moves, can't expand
    return

  node.num_children = valid_moves

  let state = node.state
  var child_idx = 0

  for move in 0..<9:
    let tile = state.board[move]

    if tile != '.':
      continue

    let symbol = state.current_player_symbol

    # Create new game state with this move

    let new_state = GameStateRef()
    new_state[] = state[]

    new_state.board[move] = symbol
    new_state.advance_player()

    # Create child node

    node.children.add(new_state.create_mcts_node(move))
    child_idx.inc

  node.expanded = true

# Calculating the UCB score

proc calculate_ucb(node: MCTSNodeRef, parent_visits: int): float =

  if node.visits == 0:
    return LARGE_MAX_VALUE

  # UCB formula: exploitation + exploration

  let exploitation = node.score / node.visits.float

  let exploration = UCB_CONSTANT * sqrt(ln(parent_visits.float) / node.visits.float)

  return exploitation + exploration

# Select the best chlid node accoring to UCB scores

proc select_best_child(node: MCTSNodeRef): MCTSNodeRef =

  var best_score = -LARGE_MAX_VALUE

  for child in node.children:
    let ucb = child.calculate_ucb(node.visits)

    if ucb > best_score:
      best_score = ucb
      result = child

# Perform a simulation from the given state to the end of the game

proc simulate_mcts(input_state: GameStateRef): float =

  let state = GameStateRef()
  state[] = input_state[]

  var
    winner: char

  while not state.check_game_over(winner):
    let move = state.get_random_move()
    let symbol = state.current_player_symbol
    state.board[move] = symbol
    state.advance_player()

  let starting_player = state.next_player

  if winner == 'T':
    return 0.5 # Tie
  elif ((
    (winner == 'X' and starting_player == 0) or
    (winner == 'O' and starting_player == 1)
  )):
    return 1.0 # Win
  else:
    return 0.0 # Loss

proc backpropagate(node: MCTSNodeRef, score: float) =
  while not node.is_nil:
    node.visits.inc

    # Flip the score when moving up the tree since players alternate
    node.score += score

    # score = 1.0 - score
    # This implementation does not track parents
    break

# Perform one iteration of the MCTS algorithm

proc iterate(node: MCTSNodeRef) =

  var
    curr_node: MCTSNodeRef = node

  # 1. Selection: traverse the tree to find the most promising leaf nodes
  while curr_node.expanded and curr_node.num_children > 0:
    curr_node = curr_node.select_best_child()

  var
    winner: char

  let state = curr_node.state

  # Check if game is over at this node
  if state.check_game_over(winner):
    # Game is over, backpropagate the result

    let score = if winner == 'T':
      0.5 # Tie
    elif ((
      (winner == 'X' and state.current_player == 1) or
      (winner == 'O' and state.current_player == 0)
    )):
      1.0 # Win for the player who just moved
    else:
      0.0 # Loss for the player who just moved

    curr_node.backpropagate(score)
    return

  # 2. Expansion: Create all possible child nodes
  curr_node.expand()

  # If not children were created, simulate from current node
  if curr_node.num_children == 0:
    let score = state.simulate_mcts()
    curr_node.backpropagate(score)
    return

  # 3. Simulation: choose a random child and simulate
  let rand_child_idx = rand(curr_node.num_children - 1)
  let rand_child = curr_node.children[rand_child_idx]

  # 4. Simulation: play the game randomly to the end
  let score = state.simulate_mcts()

  # 5. Backpropagation: update statistics
  rand_child.visits.inc
  rand_child.score += score
  curr_node.backpropagate(1.0 - score) # Parent gets opposite score

# Get the best move using MCTS

proc get_mcts_move(state: GameStateRef): int =

  # Create root node
  let root = state.create_mcts_node(-1)

  # Run MCTS simulations
  for i in 0..<MCTS_SIMULATIONS:
    root.iterate()

  # Find the child with the highest number of visits
  var
    best_move = -1
    most_visits = -1

  # First expand the root node if not already expanded
  for child in root.children:
    if child.visits > most_visits:
      most_visits = child.visits
      best_move = child.move

  return best_move

# ReLU activation function

proc relu(x: float): float =
  if x > 0: x else: 0.0

# Derivative of ReLU activation function

proc relu_derivative(x: float): float =
  if x > 0: 1.0 else: 0.0

# Initialize a neural network with random weights, we should
# use something like He weights since we use RELU, but we don't
# care as this is a trivial example.

proc random_weight(): float =
  rand(1.0) - 0.5

proc init_neural_network(nn: NeuralNetworkRef) =

  for weight in nn.weights_ih.mitems:
    weight = random_weight()

  for weight in nn.weights_ho.mitems:
    weight = random_weight()

  for weight in nn.biases_h.mitems:
    weight = random_weight()

  for weight in nn.biases_o.mitems:
    weight = random_weight()

# Apply softmax activation function to an array input, and
# set the result into output.

proc softmax(
  input: array[NN_OUTPUT_SIZE, float],
  output: var array[NN_OUTPUT_SIZE, float]
) =

  # Find maximum value then subtact it to avoid
  # numerical stability issues with exp().

  var max_value = input[0]

  for i in 1..<input.len:
    let value = input[i]
    if (value > max_value):
      max_value = value

  # Calculate exp(x_i - max) for each element and sum.

  var sum = 0.0

  for i in 0..<input.len:
    output[i] = exp(input[i] - max_value)
    sum += output[i]

  # Normalize to get probabilities.

  assert sum > 0

  for i in 0..<output.len:
    output[i] /= sum

# Neural network foward pass (inference). We store the activations
# so we can also do backpropagation later.

proc forward_pass(nn: NeuralNetworkRef, inputs: array[NN_INPUT_SIZE, float]) =

  # Copy Inputs

  for i in 0..<NN_INPUT_SIZE:
    nn.inputs[i] = inputs[i]

  # Input to hidden layer.

  for i in 0..<NN_HIDDEN_SIZE:
    var sum = nn.biases_h[i]

    for j in 0..<NN_INPUT_SIZE:
      sum += inputs[j] * nn.weights_ih[j * NN_HIDDEN_SIZE + i]

    nn.hiddens[i] = relu(sum)

  # Hidden to output (raw logits).

  for i in 0..<NN_OUTPUT_SIZE:
    nn.raw_logits[i] = nn.biases_o[i]
    for j in 0..<NN_HIDDEN_SIZE:
      nn.raw_logits[i] += nn.hiddens[j] * nn.weights_ho[j * NN_OUTPUT_SIZE + i]

  # Apply softmax to get the final probabilities.

  softmax(nn.raw_logits, nn.outputs)

# Get the best move for the computer using the neural network.
# Note that there is no complex sampling at all, we just get
# the output with the highest value THAT has an empty tile.

proc get_computer_move(
  state: GameStateRef,
  nn: NeuralNetworkRef,
  display_probs: bool
): int =

  var inputs: array[NN_INPUT_SIZE, float]

  state.board_to_inputs(inputs)
  nn.forward_pass(inputs)

  # Find the highest probability value and best legal move.

  var
    highest_prob = -1.0
    highest_prob_idx = -1
    best_move = -1
    best_legal_prob = -1.0

  for i, output in nn.outputs.pairs:
    if output > highest_prob:
      highest_prob = output
      highest_prob_idx = i

    if (
      state.board[i] == '.' and
      (best_move == -1 or output > best_legal_prob)
    ):
      best_move = i
      best_legal_prob = output

  # That's just for debugging. It's interesting to show to user
  # in the first iterations of the game, since you can see how initially
  # the net picks illegal moves as best, and so forth.

  if display_probs:
    echo "Neural network move probabilities:\n"

    for row in 0..<3:
      for col in 0..<3:
        let pos = row * 3 + col
        echo &"{(nn.outputs[pos] * 100.0):.3f}"

        if pos == highest_prob_idx:
          echo "*"

        if pos == best_move:
          echo "#"

        echo " "

      echo "\n"

    # Sum of probabilities should be 1.0, hopefully.
    # Just debugging.

    let total_prob = nn.outputs.sum

    echo &"Sum of all probabilities: {total_prob}"

  best_move

# Backpropagation function.
# The only difference here from vanilla backprop is that we have
# a 'reward_scaling' argument that makes the output error more/less
# dramatic, so that we can adjust the weights proportionally to the
# reward we want to provide.

proc backprop(
  nn: NeuralNetworkRef,
  target_probs: array[NN_OUTPUT_SIZE, float],
  learning_rate: float,
  reward_scaling: float
) =
  var output_deltas: array[NN_OUTPUT_SIZE, float]
  var hidden_deltas: array[NN_HIDDEN_SIZE, float]

  # === STEP 1: Compute deltas === */

  # Calculate output layer deltas:
  # Note what's going on here: we are technically using softmax
  # as output function and cross entropy as loss, but we never use
  # cross entropy in practice since we check the progresses in terms
  # of winning the game.
  #
  # Still calculating the deltas in the output as:
  #
  #     output[i] - target[i]
  #
  # Is exactly what happens if you derivate the deltas with
  # softmax and cross entropy.
  #
  # LEARNING OPPORTUNITY: This is a well established and fundamental
  # result in neural networks, you may want to read more about it. */

  for i in 0..<NN_OUTPUT_SIZE:
    output_deltas[i] = (nn.outputs[i] - target_probs[i]) * abs(reward_scaling)

  for i in 0..<NN_HIDDEN_SIZE:
    var error = 0.0

    for j in 0..<NN_OUTPUT_SIZE:
      error += output_deltas[j] * nn.weights_ho[i * NN_OUTPUT_SIZE + j]

    hidden_deltas[i] = error * relu_derivative(nn.hiddens[i])

  # === STEP 2: Weights updating ===

  # Output layer weights and biases.

  for i in 0..<NN_HIDDEN_SIZE:
    for j in 0..<NN_OUTPUT_SIZE:
      nn.weights_ho[i * NN_OUTPUT_SIZE + j] -= learning_rate * output_deltas[j] * nn.hiddens[i]

  for j in 0..<NN_OUTPUT_SIZE:
    nn.biases_o[j] -= learning_rate * output_deltas[j]

  # Hidden layer weights and biases.

  for i in 0..<NN_INPUT_SIZE:
    for j in 0..<NN_HIDDEN_SIZE:
      nn.weights_ih[i * NN_HIDDEN_SIZE + j] -= learning_rate * hidden_deltas[j] * nn.inputs[i]

  for j in 0..<NN_HIDDEN_SIZE:
    nn.biases_h[j] -= learning_rate * hidden_deltas[j]

# Train the neural network based on game outcome.
#
# The move_history is just an integer array with the index of all the
# moves. This function is designed so that you can specify if the
# game was started by the move by the NN or human, but actually the
# code always let the human move first.

proc learn_from_game(
  nn: NeuralNetworkRef,
  move_history: var array[9, int],
  num_moves: int,
  nn_moves_even: bool,
  winner: char
) =

  let nn_symbol = if nn_moves_even: 'O' else: 'X'

  let reward = if winner == 'T':
    0.3 # Small reward for draw
  elif (winner == nn_symbol):
    1.0 # Large reward for win
  else:
    -2.0 # Negative reward for loss

  let state = GameStateRef()
  var target_probs: array[NN_OUTPUT_SIZE, float]

  for move_idx in 0..<num_moves:
    # Skip if this wasn't a move by the neural network.
    if (
      (nn_moves_even and (move_idx mod 2) != 1) or
      (not nn_moves_even and (move_idx mod 2) != 0)
    ):
      continue

    state.init_game()

    for i in 0..<move_idx:
      let symbol = if ((i mod 2) == 0): 'X' else: 'O'
      state.board[move_history[i]] = symbol

    # Convert board to inputs and do forward pass.

    var inputs: array[NN_INPUT_SIZE, float]
    state.board_to_inputs(inputs)
    nn.forward_pass(inputs)

    # The move that was actually made by the NN, that is
    # the one we want to reward (positively or negatively).

    let move = move_history[move_idx]

    # Here we can't really implement temporal difference in the strict
    # reinforcement learning sense, since we don't have an easy way to
    # evaluate if the current situation is better or worse than the
    # previous state in the game.
    #
    # However "time related" we do something that is very effective in
    # this case: we scale the reward according to the move time, so that
    # later moves are more impacted (the game is less open to different
    # solutions as we go forward).
    #
    # We give a fixed 0.5 importance to all the moves plus
    # a 0.5 that depends on the move position.
    #
    # NOTE: this makes A LOT of difference. Experiment with different
    # values.
    #
    # LEARNING OPPORTUNITY: Temporal Difference in Reinforcement Learning
    # is a very important result, that was worth the Turing Award in
    # 2024 to Sutton and Barto. You may want to read about it.

    let move_importance = 0.5 + 0.5 * move_idx.float / num_moves.float
    let scaled_reward = reward * move_importance

    # Create target probability distribution:
    # let's start with the logits all set to 0.

    for i in 0..<NN_OUTPUT_SIZE:
      target_probs[i] = 0

    # Set the target for the chosen move based on reward:

    if (scaled_reward >= 0):
      # For positive reward, set probability of the chosen move to
      # 1, with all the rest set to 0.
      target_probs[move] = 1
    else:
      # For negative reward, distribute probability to OTHER
      # valid moves, which is conceptually the same as discouraging
      # the move that we want to discourage.
      let valid_moves_left = 9 - move_idx - 1
      let other_prob = 1.0 / valid_moves_left.float
      for i in 0..<9:
        if (
          (state.board[i] == '.') and
          (i != move)
        ):
          target_probs[i] = other_prob

    nn.backprop(target_probs, LEARNING_RATE, scaled_reward)

# Play one game of Tic Tac Toe against the neural network.

proc play_game(nn: NeuralNetworkRef) =
  let state = GameStateRef()

  var
    winner: char
    move_history: array[9, int]
    num_moves: int

  state.init_game()

  echo "Welcome to Tic Tac Toe! You are X, the computer is O.\n"
  echo "Enter positions as numbers from 0 to 8 (see picture).\n"

  while not state.check_game_over(winner):
    state.display_board()

    if state.current_player == 0:
      # Human turn.
      var
        move: int
        movec: char

      echo "Your move (0-8): "

      movec = getch()
      move = movec.ord - '0'.ord

      if (move == 9):
        quit()

      if (
        (move < 0) or
        (move > 8) or
        (state.board[move] != '.')
      ):
        echo "Invalid move! Try again.\n"
        continue

      state.board[move] = 'X'

      move_history[num_moves] = move
      num_moves.inc

    else:
      # Computer's turn

      echo "Computer's move"
      let move = state.get_computer_move(nn, true)
      state.board[move] = 'O'
      echo &"Computer placed O at position {move}\n"

      move_history[num_moves] = move
      num_moves.inc

    state.advance_player()

  state.display_board()

  if (winner == 'X'):
    echo "You win!\n"
  elif (winner == 'O'):
    echo "Computer wins!\n"
  else:
    echo "It's a tie!\n"

  nn.learn_from_game(move_history, num_moves, true, winner)

# Play a game against MCTS moves and learn from it.
#
# This is a very simple Montecarlo Method applied to reinforcement
# learning:
#
# 1. We play a complete random game (episode).
# 2. We determine the reward based on the outcome of the game.
# 3. We update the neural network in order to maximize future rewards.

proc play_mcts_game(
  nn: NeuralNetworkRef,
  move_history: var array[9, int],
  num_moves: int
): char =

  let state = GameStateRef()

  var
    winner: char
    move: int
    num_moves = 0

  state.init_game()

  while not state.check_game_over(winner):

    move = if (state.current_player == 0):
      state.get_mcts_move()
    else:
      state.get_computer_move(nn, false)

    let symbol = if (state.current_player == 0): 'X' else: 'O'

    state.board[move] = symbol

    move_history[num_moves] = move
    num_moves.inc

    state.current_player = state.current_player xor 1

  nn.learn_from_game(move_history, num_moves, true, winner)
  return winner

# Train the neural network against random moves.

proc train_against_random(
  nn : NeuralNetworkRef,
  num_games: int,
  eval_every = 1000
) =
  var
    move_history: array[9, int]
    num_moves: int
    wins = 0
    losses = 0
    ties = 0

  echo &"Training neural network against {num_games} MCTS games...\n"

  var played_games = 0

  for i in 0..<num_games:
    let winner = nn.play_mcts_game(move_history, num_moves)

    played_games.inc

    # Accumulate statistics that are provided to the user (it's fun).

    if (winner == 'O'):
      wins.inc
    elif (winner == 'X'):
      losses.inc
    else:
      ties.inc

    proc to_percent(num, den: int): float =
      num.float * 100 / den.float

    if (((i + 1) mod eval_every) == 0):
      echo &"Games: {i + 1}, Wins: {wins} ({to_percent(wins, played_games):.3f}), Losses: {losses} ({to_percent(losses, played_games):.3f}), Ties: {ties} ({to_percent(ties, played_games):.3f})"

      played_games = 0
      wins = 0
      losses = 0
      ties = 0

  echo "\nTraining complete!\n"

when is_main_module:

  var mcts_games = 20_000 # Fast and enough to play in a decent way.

  let args = command_line_params()

  if args.len > 0:
    mcts_games = parse_int(args[0])

  # Initialize neural network.

  let nn = NeuralNetworkRef()

  nn.init_neural_network()

  if mcts_games > 0:
    nn.train_against_random(mcts_games)

  var play_again: char

  while true:
    nn.play_game()

    echo "Play again? (y/n): "
    let input_char = getch()

    if (input_char != 'y'  and play_again != 'Y'):
      break
