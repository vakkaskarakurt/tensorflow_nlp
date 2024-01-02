import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or '3' to suppress all messages

from packaging import version

if version.parse(tf.__version__) < version.parse('2.0'):
    raise Exception('This notebook is compatible with TensorFlow 2.0 or higher.')

# Use os.path to construct the file path
SHAKESPEARE_TXT = os.path.join(os.path.dirname(__file__), 'shakespeare.txt')

def transform(txt):
    return np.asarray([ord(c) for c in txt if ord(c) < 255], dtype=np.int32)

def input_fn(seq_len=100, batch_size=1024):
    with tf.io.gfile.GFile(SHAKESPEARE_TXT, 'r') as f:
        txt = f.read()

    source = tf.constant(transform(txt), dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices(source).batch(seq_len + 1, drop_remainder=True)

    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    BUFFER_SIZE = 10000
    ds = ds.map(split_input_target).shuffle(BUFFER_SIZE).batch(batch_size, drop_remainder=True)

    return ds.repeat()

EMBEDDING_DIM = 512

def lstm_model(seq_len=100, batch_size=None, stateful=True):
    source = tf.keras.Input(
        name='seed', shape=(seq_len,), batch_size=batch_size, dtype=tf.int32)

    embedding = tf.keras.layers.Embedding(input_dim=256, output_dim=EMBEDDING_DIM)(source)
    lstm_1 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(embedding)
    lstm_2 = tf.keras.layers.LSTM(EMBEDDING_DIM, stateful=stateful, return_sequences=True)(lstm_1)
    predicted_char = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(256, activation='softmax'))(lstm_2)
    return tf.keras.Model(inputs=[source], outputs=[predicted_char])

tf.keras.backend.clear_session()

strategy = tf.distribute.MirroredStrategy()  # Use MirroredStrategy for GPU

with strategy.scope():
    training_model = lstm_model(seq_len=100, stateful=False)
    training_model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'])

training_model.fit(
    input_fn(),
    steps_per_epoch=100,
    epochs=10
)
training_model.save_weights('/tmp/bard.h5', overwrite=True)

BATCH_SIZE = 5
PREDICT_LEN = 250

prediction_model = lstm_model(seq_len=1, batch_size=BATCH_SIZE, stateful=True)
prediction_model.load_weights('/tmp/bard.h5')

seed_txt = 'Looks it not like the king?  Verily, we must go! '
seed = transform(seed_txt)
seed = np.repeat(np.expand_dims(seed, 0), BATCH_SIZE, axis=0)

prediction_model.reset_states()
for i in range(len(seed_txt) - 1):
    prediction_model.predict(seed[:, i:i + 1])

predictions = [seed[:, -1:]]
for i in range(PREDICT_LEN):
    last_word = predictions[-1]
    next_probits = prediction_model.predict(last_word)[:, 0, :]
    
    next_idx = [
        np.random.choice(256, p=next_probits[i])
        for i in range(BATCH_SIZE)
    ]
    predictions.append(np.asarray(next_idx, dtype=np.int32))

for i in range(BATCH_SIZE):
    print('PREDICTION %d\n\n' % i)
    p = [predictions[j][i] for j in range(PREDICT_LEN)]
    generated = ''.join([chr(c) for c in p])
    print(generated)
    print()
    assert len(generated) == PREDICT_LEN, 'Generated text too short'
