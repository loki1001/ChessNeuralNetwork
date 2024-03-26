import random
import time
import tkinter as tk
import tkinter.messagebox as messagebox

import chess
import chess.engine
import numpy as np

import tensorflow.keras.callbacks as callbacks
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import tensorflow.keras.optimizers as optimizers
from keras.models import load_model

# Constants
BOARD_SIZE = 8
SQUARE_SIZE = 60
COLOR_LIGHT_SQUARE = "#FFCE9E"
COLOR_DARK_SQUARE = "#D18B47"


# Function to generate a random board
def random_board_generator(max_depth):
    chess_board = chess.Board()  # Initialize chess board
    depth = random.randrange(0, max_depth)  # Randomly select depth within the range
    for i in range(depth):
        all_moves = list(chess_board.legal_moves)  # Get all legal moves for the current board position
        random_move = random.choice(all_moves)  # Choose a random move from the legal moves
        chess_board.push(random_move)  # Apply the chosen move to the board
        if chess_board.is_game_over():  # If the game is over, stop generating moves
            break
    return chess_board  # Return the generated chess board


# Function to evaluate a board position using Stockfish
def stockfish_evaluation(board, depth):
    # Open connection to Stockfish
    with chess.engine.SimpleEngine.popen_uci('/usr/local/Cellar/stockfish/16.1/bin/stockfish') as stockfish:
        # Analyze the board position with the specified depth limit
        result = stockfish.analyse(board, chess.engine.Limit(depth=depth))
        # Extract the evaluation score from the analysis
        score = result['score'].black().score()
    return score  # Return the evaluation score


# Dictionary to map the board positions
board_positions = {
    'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7
}


# Function to convert square to index
def square_to_index(square):
    letter = chess.square_name(square)  # Get algebraic notation of the square
    row = 8 - int(letter[1]) # Convert rank to row index
    column = board_positions[letter[0]]  # Map file to column index using board_positions dictionary
    return row, column  # Return row and column indices


# Function to convert board to matrix representation
def board_to_matrix(board):
    # Initialize a 3D numpy array to represent the board
    # 12 dimensions -> different chess piece types (Pawn, Knight, Bishop, Rook, Queen, King) for white and black
    # +2 dimensions -> legal moves for current player and opponent
    board_3d = np.zeros((14, 8, 8), dtype=np.int8)

    # Iterate over each piece type
    for piece in chess.PIECE_TYPES:
        # Iterate over each white piece on the board
        for square in board.pieces(piece, chess.WHITE):
            # Convert square index to row and column indices and mark the corresponding position in the matrix with 1
            index = np.unravel_index(square, (8, 8))
            # piece - 1 -> layer of 3D array corresponding to piece type (0-5)
            # 7 - index[0] -> row index (row numbering in chess module is reversed)
            # index[1] -> column index
            board_3d[piece - 1][7 - index[0]][index[1]] = 1

        # Iterate over each black piece on the board
        for square in board.pieces(piece, chess.BLACK):
            # Convert square index to row and column indices, and mark the corresponding position in the matrix with 1
            index = np.unravel_index(square, (8, 8))
            # piece + 5 -> layer of 3D array corresponding to black piece type (6-11)
            # 7 - index[0] -> row index (row numbering in chess module is reversed)
            # index[1] -> column index
            board_3d[piece + 5][7 - index[0]][index[1]] = 1

    # Store legal moves for the current player
    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board_3d[12][i][j] = 1  # Layer 12

    # Store legal moves for the opponent player
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board_3d[13][i][j] = 1  # Layer 13

    # Restore the original player turn
    board.turn = aux

    return board_3d  # Return the 3D matrix representation of the board


# Function to populate dataset with board positions and evaluations
def populate_dataset(size):
    # Initialize lists to store board positions (b) and evaluations (v)
    b = []
    v = []

    # Generate samples for the dataset
    for i in range(size):
        # Generate a random chess board with maximum depth of 200 moves
        new_board = random_board_generator(200)

        # Convert the board to its matrix representation
        matrix_representation = board_to_matrix(new_board)

        # Evaluate the board using Stockfish engine with a depth limit of 10
        valuation = stockfish_evaluation(new_board, 10)

        # Skip if the evaluation is not available
        if valuation is None:
            continue

        # Add the matrix representation of the board and its evaluation to the lists
        b.append(matrix_representation)
        v.append(valuation)

    # Convert the lists to numpy arrays
    b = np.array(b)
    v = np.array(v)

    # Save the dataset as a compressed .npz file
    np.savez('dataset', b=b, v=v)


# Function to build convolutional neural network model
def build_conv_model(conv_size, conv_depth):
    # Define the input layer for the 3D matrix representation of the chess board
    # 12 dimensions -> different chess piece types (Pawn, Knight, Bishop, Rook, Queen, King) for white and black
    # +2 dimensions -> legal moves for current player and opponent
    board_3d = layers.Input(shape=(14, 8, 8))

    #  Set the input as the initial value of x
    x = board_3d

    # Add convolutional layers to the model based on the conv_depth, more -> deeper
    for i in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu')(x)

    # Flatten the output of the convolutional layers to prepare processing dense layers
    x = layers.Flatten()(x)

    # Add dense layers to the model for further processing
    # Connect everything together using 64 neurons
    x = layers.Dense(64, 'relu')(x)
    # 1 neuron for 0-1 range output
    x = layers.Dense(1, 'sigmoid')(x)

    # Define the model with input and output layers
    return models.Model(inputs=board_3d, outputs=x)


# Function to get dataset from saved file
def get_dataset():
    # Load the dataset from the saved .npz file
    container = np.load('dataset.npz')

    # Extract board positions (b) and evaluations (v) from the loaded container
    b, v = container['b'], container['v']

    # Normalize the evaluation scores to the range [0, 1]
    v = np.asarray(v / abs(v).max() / 2 + 0.5, dtype=np.float32)

    return b, v  # Return the board positions and evaluations


# Function to train the neural network model
def train_model(model, x_train, y_train):
    # Compile the model with Adam optimizer and mean squared error loss -> regression
    model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')

    # Print a summary of the model architecture
    model.summary()

    # Train the model with the specified training data
    model.fit(
        x_train, y_train,  # Training data
        batch_size=2048,  # Samples before weights updated
        epochs=2,  # Train over dataset twice
        verbose=1,  # Print progress
        validation_split=0.1,  # 10% for data validation
        callbacks=[
            callbacks.ReduceLROnPlateau(monitor='loss', patience=10),  # Reduce learning when loss stops improving in 10 epochs
            callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)  # Stop training when loss doesn't decrease by at least 0.0001
        ]
    )

    # Save the trained model
    model.save('chess_model.h5')


# Function to evaluate a board position using minimax algorithm
def minimax_eval(board, model):
    # Convert the board to its 3D matrix representation and add a batch dimension for format
    board3d = board_to_matrix(board)
    board3d = np.expand_dims(board3d, 0)

    # Predict the evaluation score using the neural network model
    return model.predict(board3d)[0][0]


# Minimax algorithm implementation
def minimax(board, depth, alpha, beta, maximizing_player, model):
    # Base case: if the maximum depth is reached or the game is over, return the evaluation score
    if depth == 0 or board.is_game_over():
        return minimax_eval(board, model)

    # If it's the turn of the maximizing player
    if maximizing_player:
        max_eval = -np.inf
        # Iterate over legal moves and evaluate each possible resulting board position
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False, model)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            # Perform alpha-beta pruning
            if beta <= alpha:
                break
        return max_eval
    # If it's the turn of the minimizing player
    else:
        min_eval = np.inf
        # Iterate over legal moves and evaluate each possible resulting board position
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True, model)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            # Perform alpha-beta pruning
            if beta <= alpha:
                break
        return min_eval


# Function to get the AI's move using minimax algorithm
def get_ai_move(board, depth, model):
    max_move = None
    max_eval = -np.inf

    # Iterate over legal moves and evaluate each possible resulting board position
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, -np.inf, np.inf, False, model)
        board.pop()
        # Keep track of the move with the maximum evaluation score
        if eval > max_eval:
            max_eval = eval
            max_move = move

    return max_move  # Return the AI's chosen move


# Variable to hold label for displaying AI move calculation status
calculating_label = None


# Function to create the chessboard GUI
def create_chessboard(board):
    # Initialize the Tkinter window
    root = tk.Tk()
    root.title("Chess Game")

    # Create a canvas for drawing the chessboard
    canvas = tk.Canvas(root, width=SQUARE_SIZE * BOARD_SIZE + 40, height=SQUARE_SIZE * BOARD_SIZE + 40)
    canvas.pack()

    # List to keep track of highlighted squares
    highlighted_squares = []

    # Function to highlight a square on the chessboard
    def highlight_square(row, col, color):
        x0, y0 = col * SQUARE_SIZE + 20, row * SQUARE_SIZE + 20
        x1, y1 = x0 + SQUARE_SIZE, y0 + SQUARE_SIZE
        canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=4)
        highlighted_squares.append((x0, y0, x1, y1))

    # Function to draw the chessboard
    def draw_chessboard():
        outer_padding = 20

        # Draw squares and pieces on the chessboard
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                color = COLOR_LIGHT_SQUARE if (row + col) % 2 == 0 else COLOR_DARK_SQUARE
                x0 = outer_padding + col * (SQUARE_SIZE)
                y0 = outer_padding + row * (SQUARE_SIZE)
                x1 = x0 + SQUARE_SIZE
                y1 = y0 + SQUARE_SIZE
                canvas.create_rectangle(x0, y0, x1, y1, fill=color)

                piece = board.piece_at(chess.square(col, 7 - row))
                if piece is not None:
                    color = "white" if piece.color == chess.WHITE else "black"
                    text = piece.unicode_symbol()
                    font = ("Helvetica", 32)
                    text_x = x0 + SQUARE_SIZE // 2
                    text_y = y0 + SQUARE_SIZE // 2
                    canvas.create_text(text_x, text_y, text=text, fill=color, font=font)

        # Draw row and column labels
        for i in range(BOARD_SIZE):
            x = outer_padding // 2
            y = outer_padding + (BOARD_SIZE - i - 1) * (SQUARE_SIZE) + SQUARE_SIZE // 2
            canvas.create_text(x, y, text=str(i + 1), fill="#FEF8F2")

        for i in range(BOARD_SIZE):
            x = outer_padding + BOARD_SIZE * SQUARE_SIZE + outer_padding // 2
            y = outer_padding + (BOARD_SIZE - i - 1) * (SQUARE_SIZE) + SQUARE_SIZE // 2
            canvas.create_text(x, y, text=str(i + 1), fill="#FEF8F2")

        for i, letter in enumerate("abcdefgh"):
            x = outer_padding + i * (SQUARE_SIZE) + SQUARE_SIZE // 2
            y = outer_padding // 2
            canvas.create_text(x, y, text=letter, fill="#FEF8F2")

        for i, letter in enumerate("abcdefgh"):
            x = outer_padding + i * (SQUARE_SIZE) + SQUARE_SIZE // 2
            y = outer_padding + BOARD_SIZE * SQUARE_SIZE + outer_padding // 2
            canvas.create_text(x, y, text=letter, fill="#FEF8F2")

    # Function to handle user move input
    def handle_user_move():
        global calculating_label
        user_move = entry.get()
        clear_highlight()
        update_board_display()
        try:
            move = chess.Move.from_uci(user_move)
            if move in board.legal_moves:
                board.push(move)
                update_board_display()

                # Check game status after the user move
                if board.is_checkmate():
                    messagebox.showinfo("Game Over", "Checkmate! You've won the game.")
                elif board.is_check():
                    messagebox.showinfo("Check", "AI in check!")

                # Update label to indicate AI is calculating move
                if calculating_label:
                    calculating_label.config(text="AI is calculating...")
                else:
                    calculating_label = tk.Label(root, text="AI is calculating...")
                    calculating_label.pack()

                # Calculate AI move and update board display
                start_time = time.time()
                canvas.after(50, lambda: [
                    board.push(get_ai_move(board, 2, model)),
                    update_board_display(),
                    messagebox.showinfo("Game Over",
                                        "Checkmate! You've lost the game.") if board.is_checkmate() else None,
                    messagebox.showinfo("Check", "You are in check.") if board.is_check() else None,
                    calculating_label.config(text=f"AI move calculated in {round(time.time() - start_time, 2)} seconds")
                ])

                update_board_display()
            else:
                messagebox.showwarning("Invalid Move", "Invalid move. Please try again.")
                highlight_invalid_move(user_move)
        except ValueError:
            messagebox.showerror("Invalid Move Format",
                                 "Invalid move format. Please enter a move in UCI notation (e.g., e2e4).")
        entry.delete(0, tk.END)

    # Function to highlight an invalid move on the chessboard
    def highlight_invalid_move(user_move):
        try:
            move = chess.Move.from_uci(user_move)
            from_square = move.from_square
            to_square = move.to_square
            from_row, from_col = 7 - from_square // 8, from_square % 8
            to_row, to_col = 7 - to_square // 8, to_square % 8
            highlight_square(from_row, from_col, "red")
            highlight_square(to_row, to_col, "red")
        except ValueError:
            messagebox.showerror("Invalid Move Format",
                                 "Invalid move format. Please enter a move in UCI notation (e.g., e2e4).")

    # Function to clear highlighted squares on the chessboard
    def clear_highlight():
        for square in highlighted_squares:
            canvas.create_rectangle(square, fill="", outline="")
        del highlighted_squares[:]

    # Function to update the chessboard display
    def update_board_display():
        canvas.delete("all")
        draw_chessboard()

    # Label and Entry for user move input
    label = tk.Label(root, text="Enter your move (e.g., e2e4):")
    label.pack()

    entry = tk.Entry(root)
    entry.pack()

    # Button to trigger user move handling
    button = tk.Button(root, text="Make Move", command=handle_user_move)
    button.pack()

    # Draw the initial chessboard
    draw_chessboard()

    # Start Tkinter event loop
    root.mainloop()


if __name__ == "__main__":
    # Check if a pre-trained model exists
    try:
        model = load_model('chess_model.h5')
        print("Using existing model for game.")
    # If no pre-trained model found, train a new one
    except FileNotFoundError:
        print("Model not found. Training a new model...")
        # Generate dataset for training
        populate_dataset(1000)
        x_train, y_train = get_dataset()
        # Build and train a convolutional neural network model
        model = build_conv_model(32, 4)
        train_model(model, x_train, y_train)

    # Start the game by initializing the chessboard
    board = chess.Board()
    create_chessboard(board)
