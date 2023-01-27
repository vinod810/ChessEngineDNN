import sys

import chess
import chess.engine
from board_evaluator import evaluate_regular, evaluate_ml, material_value

class console_colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

Full_Depth = 2
Q_Depth = 12

def _determine_best_move(board, is_white, current_depth = 0, pv=""):
    """Given a board, determines the best move.

    Args:
        is_white (bool): Whether the particular move is for white or black.
    """

    best_value = -100000 if is_white else 100000
    best_move = None
    material = material_value(board)
    max_depth = current_depth # TODO Max_Depth in PerfMeas
    best_pv = None
    for move in board.legal_moves:
        board.push(move)
        pv_new = pv + str(move)
        material_new = material_value(board)
        value, depth, pv_new = _minimax_helper(current_depth + 1, board, -10000, 10000, not is_white, material_new, pv_new)
        print(pv_new, value, depth)
        board.pop()
        max_depth = max(max_depth, depth)
        if (is_white and value > best_value) or (not is_white and value < best_value):
            best_value = value
            best_move = move
            best_pv = pv_new
    return best_move, best_value, max_depth, best_pv


def _minimax_helper(current_depth, board, alpha, beta, is_maximizing, material, pv):

    if board.is_game_over():
        return -10000 if is_maximizing else 10000, current_depth, pv

    if current_depth >= Q_Depth :
        return evaluate_regular(board, pv), current_depth, pv

    max_depth = current_depth

    if is_maximizing:
        best_value = -100000
        best_pv = None
        for move in board.legal_moves:
            board.push(move)
            material_new = material_value(board)
            pv_new = pv + str(move)
            if material == material_new and current_depth + 1 >= Full_Depth:
                value = evaluate_ml(board, pv_new)
                depth = current_depth + 1
            else:
                value, depth, pv_new = _minimax_helper(current_depth + 1, board, alpha, beta, False, material_new, pv_new)
            board.pop()
            max_depth = max(max_depth, depth)
            best_pv = pv_new if value > best_value else best_pv
            best_value = max(best_value, value)
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
        return best_value, max_depth, best_pv
    else:
        best_value = 100000
        best_pv = None
        for move in board.legal_moves:
            board.push(move)
            material_new = material_value(board)
            pv_new = pv + str(move)
            if material == material_new and current_depth + 1 >= Full_Depth:
                value = evaluate_ml(board, pv_new)
                depth = current_depth + 1
                # if pv_new == "g4e2g2c6a7a5":
                #     print(depth, value, flush=True)
                    #exit()
            else:
                value, depth, pv_new = _minimax_helper(current_depth + 1, board, alpha, beta, True, material_new, pv_new)
                # if pv == "g4e2g2c6":
                #      print("minmiax", value, depth, pv_new)
            board.pop()
            max_depth = max(max_depth, depth)
            best_pv = pv_new if value < best_value else best_pv # Watch out the comparison
            best_value = min(best_value, value)
            # if pv == "g4e2g2c6":
            #     print('best_pv=', best_pv, best_value)
            beta = min(beta, best_value)
            if beta <= alpha:
                # if pv == "g4e2g2c6":
                #      print("pruning")
                break
        return best_value, max_depth, best_pv


def main_logic(fen):
    if fen is None:
        board = chess.Board()
    else:
        board = chess.Board(fen)

    if fen is None:
        is_white = input(console_colors.OKBLUE + 'Will you be playing as white or black (white/black)? ' +
                         console_colors.ENDC).lower()[0] == "w"
    else:
        is_white = True if board.turn == chess.BLACK else False

    print()
    print(console_colors.HEADER + f'= Board State =' + console_colors.ENDC)
    print(board)

    first_iter = True
    if is_white:
        while not board.is_game_over():
            if not first_iter or fen is None:
                print()
                move = None
                while True:
                    try:
                        move = board.parse_san(input(console_colors.OKGREEN + 'Enter your move: ' + console_colors.ENDC))
                    except ValueError:
                        print(console_colors.FAIL + f'That is not a valid move!' + console_colors.ENDC)
                        continue
                    except KeyboardInterrupt:
                        exit()
                    break
                assert (move is not None)
                board.push(move)

            first_iter = False

            move, value, depth, pv = _determine_best_move(board, False)
            board.push(move)

            print()
            print(console_colors.FAIL + f'Black made the move: {move}' + console_colors.ENDC)
            print()
            print(console_colors.HEADER + f'= Board State =' + console_colors.ENDC)
            print(board)
            print("\nvalue=", value, ", depth=", depth, " pv=", pv)
            print()
    else:
        while not board.is_game_over():
            move, value, depth, pv = _determine_best_move(board, True)
            board.push(move)

            first_iter = False # FIXME
            print()
            print(console_colors.FAIL + f'White made the move: {move}' + console_colors.ENDC)
            print()
            print(console_colors.HEADER + f'= Board State =' + console_colors.ENDC)
            print(board)
            print("\nvalue=", value, ", depth=", depth, " pv=", pv)
            print()
            while True:
                try:
                    move = board.parse_san(input(console_colors.OKGREEN + 'Enter your move: ' + console_colors.ENDC))
                except ValueError:
                    print(console_colors.FAIL + f'That is not a valid move!' + console_colors.ENDC)
                    continue
                except KeyboardInterrupt:
                    exit()
                break
            board.push(move)

    print(console_colors.HEADER + f'The game is over!' + console_colors.ENDC)

if __name__ == '__main__':
    arg1 = sys.argv[1] if len(sys.argv)  == 2 else None
    main_logic(arg1)

