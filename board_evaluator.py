import chess
from board_pieces_tables import *
import tensorflow as tf
Max_Score = 600
model = tf.keras.models.load_model('../model')

def get_board_representation(board, perspective):
    colors = (chess.WHITE, chess.BLACK)
    pieces = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    if perspective == chess.BLACK:
        pieces.reverse()

    board_repr = [0] * (64 * len(pieces) * len(colors))
    base_square = 0

    for color in colors:
        for piece in pieces:
            squares = board.pieces(piece, color)
            for square in squares:
                board_repr[base_square + square] = 1
            base_square += 64

    if perspective == chess.BLACK:
        board_repr.reverse()

    return board_repr #bytearray(board_repr)


def evaluate_ml(board, pv):
    if board.is_checkmate():
        return -1000 if board.turn() == chess.WHITE else  1000

    board_repr = get_board_representation(board, board.turn)
    score = model.predict([board_repr], verbose=0, workers=16, use_multiprocessing=True)
    score = round((score[0][0] * 2.0 * Max_Score - Max_Score) / 100, 1)
    score = -score if board.turn == chess.BLACK else score
    if pv == "g4e2g2c6a7a5":
        print(score * 100, flush=True)
    return score * 100


def material_value(board):
    wp = len(board.pieces(chess.PAWN, chess.WHITE))
    bp = len(board.pieces(chess.PAWN, chess.BLACK))
    wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
    bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
    wb = len(board.pieces(chess.BISHOP, chess.WHITE))
    bb = len(board.pieces(chess.BISHOP, chess.BLACK))
    wr = len(board.pieces(chess.ROOK, chess.WHITE))
    br = len(board.pieces(chess.ROOK, chess.BLACK))
    wq = len(board.pieces(chess.QUEEN, chess.WHITE))
    bq = len(board.pieces(chess.QUEEN, chess.BLACK))

    return 100 * (wp - bp) + 300 * (wn - bn) + 300 * (wb - bb) + 500 * (wr - br) + 900 * (wq - bq)


def evaluate_regular(board, pv):
    """
    Given a particular board, evaluates it and gives it a score.
    A higher score indicates it is better for white.
    A lower score indicates it is better for black.

    Args:
        board (chess.Board): A chess board.

    Returns:
        int: A score indicating the state of the board (higher is good for white, lower is good for black)
    """
    if board.is_checkmate():
        return -1000 if board.turn() == chess.WHITE else  1000

    material = material_value(board)

    pawn_sum = sum([PAWN_TABLE[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
    pawn_sum = pawn_sum + sum([-PAWN_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.PAWN, chess.BLACK)])
    knight_sum = sum([KNIGHTS_TABLE[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
    knight_sum = knight_sum + sum([-KNIGHTS_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.KNIGHT, chess.BLACK)])
    bishop_sum = sum([BISHOPS_TABLE[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
    bishop_sum = bishop_sum + sum([-BISHOPS_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.BISHOP, chess.BLACK)])
    rook_sum = sum([ROOKS_TABLE[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
    rook_sum = rook_sum + sum([-ROOKS_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.ROOK, chess.BLACK)])
    queens_sum = sum([QUEENS_TABLE[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
    queens_sum = queens_sum + sum([-QUEENS_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.QUEEN, chess.BLACK)])
    kings_sum = sum([KINGS_TABLE[i] for i in board.pieces(chess.KING, chess.WHITE)])
    kings_sum = kings_sum + sum([-KINGS_TABLE[chess.square_mirror(i)] for i in board.pieces(chess.KING, chess.BLACK)])

    board_value = material + pawn_sum + knight_sum + bishop_sum + rook_sum + queens_sum + kings_sum

    #if pv == "g4e2g2c6b8c8":
        #print("material, board_value", material, board_value, flush=True)
    #return material # TODO FIX ME
    return board_value
