import sys
import chess.pgn

Train_Data_Factor = 0.9
Max_Score = 600 # http://talkchess.com/forum3/download/file.php?id=869
Board_Size = 8 * 8 * 6 * 2 * 2

def get_board_representation(board, perspective):
    colors = (chess.WHITE, chess.BLACK)
    pieces = [chess.KING, chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
    if perspective == chess.BLACK:
        pieces.reverse()

    board_repr = [0] * Board_Size
    base_square = 0

    for color in colors:
        for piece in pieces:
            squares = board.pieces(piece, color)
            # piece positions
            for square in squares:
                board_repr[base_square + square] = 1
                # positions attacked by the piece
                for target in board.attacks(square):
                    board_repr[base_square + 64 + target] += 1
                # TODO Add BLACK support
            base_square += 64 * 2

    if perspective == chess.BLACK:
        board_repr.reverse()

    return board_repr


def print_board_representation(board_repr, perspective):
    print("perspective=", 'White' if perspective == chess.WHITE else 'Black')
    for layer in range(24):
        board = board_repr[layer * 64: layer * 64 + 64]
        for rows in reversed(range(0, 8)):
            for columns in range(0, 8):
                print(board[rows*8 + columns], end =" ")
            print('')
        print('---------------------------')


def get_score(node, perspective):
    score = 0
    if node.eval().is_mate():
        if perspective == chess.WHITE:
            if node.eval().white().mate() > 0:
                score = Max_Score
            elif node.eval().white().mate() < 0:
                score = -Max_Score
            else:
                assert "Should not be here"
        else:
            if node.eval().black().mate() > 0:
                score = Max_Score
            elif node.eval().black().mate() < 0:
                score = -Max_Score
            else:
                assert "Should not be here"
    else:
        if perspective == chess.WHITE:
            score = node.eval().white().score()
        else:
            score = node.eval().black().score()

    score = min(score, Max_Score)
    score = max(score, -Max_Score)

    return score


def  main(pgn_file, data_dir, target_items):
    items_count = 0
    progress_step = target_items / 100
    target_train_items = int(target_items * Train_Data_Factor)

    train_data_file = data_dir + '/' + 'train_data_' + str(target_items) + '.dat'
    val_data_file = data_dir + '/' + 'val_data_' + str(target_items) + '.dat'
    train_label_file = data_dir + '/' + 'train_label_' + str(target_items) + '.dat'
    val_label_file = data_dir + '/' + 'val_label_' + str(target_items) + '.dat'

    pgn = open(pgn_file)

    with open(train_label_file, 'wb') as f_train_label, open(train_data_file, 'wb') as f_train_data, \
            open(val_label_file, 'wb') as f_val_label, open(val_data_file, 'wb') as f_val_data:

        while (game := chess.pgn.read_game(pgn)) is not None and items_count < target_items:
            # TODO Filter Variant  game.headers['Site'])
            # len(board.attackers(chess.WHITE, chess.E1))
            node = game
            while (node := node.next()) is not None:
                if node.eval() is None:
                    break

                if node.turn() == chess.WHITE:
                    board_repr = get_board_representation(node.board(), chess.WHITE)
                    #print_board_representation(board_repr, chess.WHITE)
                    #exit()
                    score = get_score(node, chess.WHITE)
                else:
                    #  TODO Re-enable Black
                    continue
                    #board_repr = get_board_representation(node.board(), chess.BLACK)
                    #print_board_representation(board_repr, chess.BLACK)
                    #score = get_score(node, chess.BLACK)

                #score = score + Max_Score # range 0 .. 2*Max_Score
                #assert(score >= 0)
                if items_count < target_train_items:
                    f_train_label.write(score.to_bytes(4, byteorder='little', signed=True))
                else:
                    f_val_label.write(score.to_bytes(4, byteorder='little', signed=True))

                byte_array = bytearray(board_repr)
                if items_count < target_train_items:
                    f_train_data.write(byte_array)
                else:
                    f_val_data.write(byte_array)

                items_count += 1
                if items_count % progress_step == 0:
                    print(int(items_count / progress_step), end=".", flush=True)

    print("\nitems_count:", items_count)
    if items_count < target_items:
        print("Warning: insufficient positions")

# zstdcat ~/Downloads/lichess_db_standard_rated_2022-11.pgn.zst | python3 pgn2data.py /dev/stdin data 100000

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 pgn2data.py pgn-file, data-dir, number-positions")
        exit()
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))




