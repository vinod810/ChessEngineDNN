import sys
import tensorflow as tf
from pgn2data import Train_Data_Factor, Max_Score, Board_Size

BATCH_SIZE = 1024
EPOCHS = 10 # 10
AUTO = tf.data.experimental.AUTOTUNE

def read_data(tf_bytestring):
    board = tf.io.decode_raw(tf_bytestring, tf.uint8)
    board = tf.reshape(board, [Board_Size])
    return board

def read_label(tf_bytestring):
    score = tf.io.decode_raw(tf_bytestring, tf.int32)
    score = (tf.cast(score, tf.float32) + Max_Score) / (2 * Max_Score) # normalize 0.0 .. 1.0
    #score = tf.reshape(score, [1])
    return score

def load_dataset(data_file, label_file):
    data_dataset = tf.data.FixedLengthRecordDataset(filenames=[data_file],
        record_bytes=Board_Size, header_bytes=0, footer_bytes=0)
    data_dataset = data_dataset.map(read_data, num_parallel_calls=8)

    label_dataset = tf.data.FixedLengthRecordDataset(filenames=[label_file],
        record_bytes=4, header_bytes=0, footer_bytes=0)
    label_dataset = label_dataset.map(read_label, num_parallel_calls=8)
    #for label in label_dataset:
        #print(label)

    dataset = tf.data.Dataset.zip((data_dataset, label_dataset))
    return dataset

def make_training_dataset(data_file, label_file):
    dataset = load_dataset(data_file, label_file)
    #dataset = dataset.cache()  # for TPU this is important
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    dataset = dataset.repeat()  # Mandatory for Keras for now
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)  # important on TPU, batch size must be fixed
    dataset = dataset.prefetch(AUTO)  # fetch next batches while training on the current one
    return dataset


def make_validation_dataset(data_file, label_file):
    dataset = load_dataset(data_file, label_file)
    #dataset = dataset.cache()
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
    dataset = dataset.repeat() # Mandatory for Keras for now
    dataset = dataset.prefetch(AUTO)
    return dataset


def  main(model_dir, data_dir, items_count):
    train_data_file = data_dir + '/' + 'train_data_' + str(items_count) + '.dat'
    val_data_file = data_dir + '/' + 'val_data_' + str(items_count) + '.dat'
    train_label_file = data_dir + '/' + 'train_label_' + str(items_count) + '.dat'
    val_label_file = data_dir + '/' + 'val_label_' + str(items_count) + '.dat'

    training_dataset = make_training_dataset(train_data_file, train_label_file)
    validation_dataset = make_validation_dataset(val_data_file, val_label_file)

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=[Board_Size, 1]),
        tf.keras.layers.Dense(1000, activation="relu"), # This layer gives only a small (0.003) improvement for 10M
        tf.keras.layers.Dense(400, activation="relu"),
        tf.keras.layers.Dense(200, activation="relu"),
        tf.keras.layers.Dense(60, activation="relu"),
        tf.keras.layers.Dense(1, activation='sigmoid'), #'linear')
    ])

    # this configures the training of the model. Keras calls it "compiling" the model.
    model.compile(
        optimizer='adam', # Adam(5e-4), #'adam',
        loss='mae', #'categorical_crossentropy',
        metrics=['mae'])  # ['accuracy'])  # % of correct answers, not used for training

    #model.summary()
    # Train the model
    checkpoint_filepath = '/tmp/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=2,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0
    )

    steps_per_epoch = int(items_count * Train_Data_Factor) // BATCH_SIZE
    print("Steps per epoch: ", steps_per_epoch)
    try:
        model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
              validation_data=validation_dataset, validation_steps=1,
                        callbacks=[model_checkpoint_callback, early_stopping_callback])
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received!")
        exit()

    model.load_weights(checkpoint_filepath)
    if items_count > 1000000:
        print('Saving model ...')
        model.save(model_dir)
    else:
        print('Model not saved')

# export TF_CPP_MIN_LOG_LEVEL=2
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 model.py model-dir data-dir items-count")
        exit()
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))




