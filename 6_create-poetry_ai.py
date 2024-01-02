from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
import numpy as np

tokenizer = Tokenizer()

# It's stored as a single string with slash n's    
# to give new lines.    
data = "In the town of Athy one Jeremy Lanigan \n Battered away ... ..."
# That, I can then break into a number of sentences    
# by splitting the string by that new line character,    
# and this will form my corpus of text.
corpus = data.lower().split("\n")

# I can then fit my tokenizer to the corpus to get a word index.
tokenizer.fit_on_texts(corpus)

# As I'm using an out of vocabulary token,    
# I'll add 1 to the length of the word index just    
# to cater for that.
total_words = len(tokenizer.word_index) + 1

# First of all, I'll create an empty list of input sequences.    
# We'll populate this as we go along.
input_sequences = []

for line in corpus:

    # Now, for each line in the corpus,    
    # we'll create the list of tokens.    
    # Note that we're not doing text to sequences    
    # for the entire body.    
    # We're going to do it one line at a time.    
    # So this will give me the text to sequences for the current line.
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        # Next, we're going to go through this list    
        # and generate n grams from that.
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# Now that we've split the sentence into multiple lists,    
# we'll need to pad it.
max_sequence_len = max([len(x) for x in input_sequences])

# So we'll start by getting the length    
# of the longest of the sentences and then    
# pad everything with a 0 up to the length    
# of the maximum sentence.
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# We can simply use code like this to generate our X's and now    
# our labels.
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]

# Finally, we'll want our Y to be categorical and one    
# hot encoded, so that when we train,    
# we'll be able to predict across all of the words in our corpus    
# which one is the most likely word to be next in the sequence    
# given the current set of words.    
# And then we can use the keras to categorical to achieve this.
ys = keras.utils.to_categorical(labels, num_classes=total_words)


from keras import Sequential
from keras.layers import Embedding, LSTM, Dense, Bidirectional
from keras.optimizers import Adam

# It starts with a sequential, adds an embedding    
# at the top like we saw earlier.
model = Sequential()
# As there's a massive variation of words,    
# I gave it a lot of dimensions.    
# And in this case, it's 240.
# The first parameter is the number    
# of unique words in the corpus.    
# The input length is the maximum sequence length minus 1,    
# because we lopped off the final value in each sequence    
# to make a label.
model.add(Embedding(total_words, 240, input_length=max_sequence_len-1))
# After that, we've just got a single LSTM,    
# but we'll make it bi-directional.    
model.add(Bidirectional(LSTM(150)))
# And then importantly, our output is a dense    
# with the total number of words.
model.add(Dense(total_words, activation='softmax'))

# Remember that the labels were 1 hot encoded,    
# so we want an output that is representative of this.    
# It's then a matter of defining your loss function    
# and optimizer.    
# Remember, as this is categorical with lots of classes,    
# you'll need a categorical loss function    
# such as categorical cross entropy here.    
# And once you've done that, you just fit the X's to the Y's.

adam = Adam(learning_rate=0.01)  # Updated for the latest version of Keras
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
history = model.fit(xs, ys, epochs=100, verbose=1)

# With the simple model architecture above,    
# it ends up with an accuracy around 70 to 75%.    
# And that means that given a sequence of words,    
# it will pick the correct word right about 70% of the time.    
# If it gets a sequence of words it hasn't previously seen,    
# it can make a rough prediction for what    
# the next word could be.

seed_text = "I made a poetry machine"
next_words = 20

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted_probabilities = model.predict(token_list, verbose=0)
    predicted_index = np.argmax(predicted_probabilities)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break
    seed_text += " " + output_word
    print(seed_text)
