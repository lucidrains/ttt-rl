# Tic Tac Toe with Reinforcement Learning

*The only winning move is not to play*

This code implements a neural network that learns to play tic-tac-toe using
reinforcement learning, just playing against a random adversary, in **under
400 lines of C code**, without any external library used. I guess there are
many examples of RL out there, written in PyTorch or with other ML frameworks,
however what I wanted to accomplish here was to have the whole thing
implemented from scratch, so that each part is understandable and self-evident.

While the code is a toy, designed to help interested people to learn the basics
of reinforcement learning, it actually showcases the power of RL in learning
things without any pre-existing clue about the game:

1. It uses cold start: the neural network is initialized with random weights.
2. *Tabula rasa* learning: no knowledge of the game is put into the program, if not the fact that X or O can't be put into already used tile, and the fact that a victory or a tie is reached when there are three same symbols in a row or all the tiles are used.
3. The only signal used to train the neural network is the reward of the game: win, lose, tie.

In my past experience with [the Kilo editor](https://github.com/antirez/kilo) and the [Picol interpreter](https://github.com/antirez/picol) I noticed that for programmers that want to understand new fields (especially young programmers) small C programs without dependencies, clearly written, commented and *very short* are a good starting point, so, in order to celebrate the Turing Award assigned to Sutton and Barto, I thought of writing this one.

To try this program, compile with:

    cc ttt.c -o ttt -O3 -Wall -W -ffast-math -lm

Then run with

    ./ttt

The [Nim](https://nim-lang.org/) version can be run with the following

```bash
$ nim c -d:release -r ttt_rl.nim
```

By default, the program plays against a random opponent (an opponent just
throwing random "X" at random places at each move) for 150k games. Then it
starts a CLI interface to play with the human user. You can specify how many
games you want it to play against the random opponent before playing with
the human specifying it as first argument:

    ./ttt 2000000

With 2 million games (a few seconds required) it usually no longer loses
a game.

After playing against itself for a few iterations, the program achieves
what is likely perfect playing:

    Games: 2000000, Wins: 1756049 (87.8%)
                    Losses: 731 (0.0%)
                    Ties: 243220 (12.2%)

Note that there are runs that are more lucky than others, likely because of
weights initialization in the neural network. Run the program multiple times
if you can't reach 0 losses.

# How it works

The code tries hard to be simple, and is well commented, with about one line of comment for each two lines of code: to understand how it works should be relatively easy. Yet, in this README, I try to outline a few important things. Also make sure to check the *LEARNING OPPORTUNITY* comments inside the code: there, I tried to highlight important results or techniques in the field of neural networks that you may want to study better.

## Board representation

The state of the game is just that:

```c
typedef struct {
    char board[9];          // Can be "." (empty) or "X", "O".
    int current_player;     // 0 for player (X), 1 for computer (O).
} GameState;
```

The human and computer play always in the same order: the human starts,
the computer replies to the move. They also play always with the same
symbol: "X" for the human, "O" for the computer.

The board itself is just represented by 9 characters, depending on the
fact the tile is empty, or contains X or O.

## Neural network

The neural network is very hard-coded, because the code really wants to be
simple: it only have a single hidden layer, which is enough to model
a so simple game (adding more layers don't help to converge faster nor
to play better).

Note that tic tac toe has only 5478 possible states, and by default with
100 hidden neurons our neural network has:

    18 (inputs) * 100 (hidden) +
    100 (hidden) * 9 (outputs) weights +
    100 + 9 biases

For a total of 2809 parameters, so our neural network is very near to be able to
memorize each state of the game. However you can reduce the hidden size
to 25 (or less) and it is still able to play well (but not perfectly) with
around 700 parameters (or less).

```c
typedef struct {
    // Weights and biases.
    float weights_ih[NN_INPUT_SIZE * NN_HIDDEN_SIZE];
    float weights_ho[NN_HIDDEN_SIZE * NN_OUTPUT_SIZE];
    float biases_h[NN_HIDDEN_SIZE];
    float biases_o[NN_OUTPUT_SIZE];

    // Activations are part of the structure itself for simplicity.
    float inputs[NN_INPUT_SIZE];
    float hidden[NN_HIDDEN_SIZE];
    float raw_logits[NN_OUTPUT_SIZE]; // Outputs before softmax().
    float outputs[NN_OUTPUT_SIZE];    // Outputs after softmax().
} NeuralNetwork;
```

Activations are always memorized directly inside the neural network,
so calculating the deltas and performing the backpropagation is very
simple.

We use RELU because of simple derivative. Almost everything would work in this
case. Weights initialization don't care about RELU, they are just random
from -0.5 to 0.5 (no He weights initialization).

The output is computed using softmax(), since this neural network basically
assigns probabilities to every possible next move. In theory we use cross
entropy to calculate the loss function, but in practice we evaluate our
*agent* based on the results of the games, so we only use it implicitly here:

```c
        deltas[i] = output[i] - target[i]
```

That is the delta in case of softmax and cross entropy.

## Reinforcement learning policy

This is the reward policy used:

```c
    if (winner == 'T') {
        reward = 0.2f;  // Small reward for draw
    } else if (winner == nn_symbol) {
        reward = 1.0f;  // Large reward for win
    } else {
        reward = -1.0f; // Negative reward for loss
    }
```

When rewarding, we create all the states of the game where the neural network was about to move, and for each state, we reward the winning moves (not just the
final move that won, but *all* the moves performed in the game we won) using as
target output all the other moves set to 0, and the move we want to reward
set to 1. Then we do a pass of backpropagation and update the weights.

For ties, it's like winning, but the reward is scaled. Instead, when the game
was lost, we use as a target the move set to 0, all the invalid moves set to
0 as well, and all the other valid moves set to `1/(number-of-valid-moves)`.

However, we also perform scaling according to how early the move was performed: for moves that are near the start of the game, we give smaller rewards, and for moves that are later in the game (near the end of the game) we provide a stronger reward:

        float move_importance = 0.5f + 0.5f * (float)move_idx/(float)num_moves;
        float scaled_reward = reward * move_importance;

Note that the above makes a lot of difference in the way the program works.
Also note that while this may seem similar to Time Difference in reinforcement
learning, it is not: we don't have a simple way in this case to evaluate if
a single step provided a positive or negative reward: we need to wait for
each game to finish. The temporal scaling above is just a way to code inside
the network that early moves are more open, while, as the game goes on, we
need to play more selectively.

## Weights updating

We just use trivial backpropagation, and the code is designed in order to
show very clearly that, after all, things work in a VERY similar way to
what happens with supervised learning: the difference is just the input/output
pairs are not known beforehand, but they are provided on the fly based on the
reward policy of reinforcement learning.

Please check the code for more information.

## Future work

Things I didn't test because the complexity would kinda sabotage the educational value of the program and/or for lack of time, but that could be interesting exercises and interesting other projects / branches:

* Can this approach work with connect four as well? The much larger space of the problem would be really interesting and less of a toy.
* Train the network to play both sides by having an additional input set, that is the symbol that is going to do the move (useful especially in the case of connect four) so that we can use the network itself as opponent, instead of playing against random moves.
* Implement proper sampling, in the case above, so that initially moves are quite random, later they start to pick more consistently the predicted move.
* MCTS.
