import sys
import chess.pgn
import tensorflow as tf

from pgn2data import get_board_representation, Max_Score


def  main(model_dir):
    model = tf.keras.models.load_model(model_dir)
    # Check its architecture
    model.summary()

    while True:
        try:
            fen = input("FEN: ")
            if fen == "":
                print("Type 'exit' to exit")
                continue
            if fen == "exit" or fen == "Exit":
                break
            board = chess.Board(fen)
            board_repr = get_board_representation(board, board.turn())
            score = model.predict([board_repr], steps=1)
            score = round((score[0][0] * 2.0 * Max_Score - Max_Score) / 100, 1)
            score = -score if board.turn() == chess.BLACK else score
            print("predicted score: ", score)
        except KeyboardInterrupt:
            if input("Type 'exit' to exit: ") != "exit":
                continue
            break


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 predict.py model-dir")
        exit()

    main(sys.argv[1])
