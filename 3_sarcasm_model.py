# First of all, we'll import the JSON library.	
import json

# Then, we can load in the sarcasm JSON	
# file using the JSON library.
with open('sarcasm.json', 'r') as f:
    datastore = json.load(f)

# We can then create lists for the labels, headlines, and article	
# URLs.
sentences = []
labels = []
urls = []

# And when we iterate through the JSON,	
# we can load the requisite values into our Python list.
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

# Now that we have three lists, one with our labels,	
# one with the text, and one with the URLs,	
# we can start doing a familiar preprocessing on the text.
    
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token="<OOV>")
# By calling tokenizer.fit on texts with the headline,	
# we'll create tokens for every word in the corpus.	

tokenizer.fit_on_texts(sentences)
# And then, we'll see them in the word index.
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post') 

# If we want to inspect them, we can simply print them out.	
# Here you can see one tokenized sentence and the shape	
# of the entire corpus.
print(padded[0]) 

# That's 26,709 sequences, each with 40 tokens.
print(padded.shape) # (26709, 40)


# Fortunately, Python makes it super easy	
# for us to slice this up.	
# Let's take a look at that next.	
# So we have a bunch of sentences in a list	
# and a bunch of labels in a list.	
# To slice them into training and test sets	
# is actually pretty easy.	
# If we pick a training size, say 20,000,	
# we can cut it up with code like this.	
# So the training sentences will be the first 20,000 sliced	
# by this syntax, and the testing sentences	
# will be the remaining slice, like this.	
# And we can do the same for the labels	
# to get a training and a test set.

training_size = 20000
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# But there's a bit of a problem.	
# Remember earlier we used the tokenizer	
# to create a word index of every word in the set?	
# That was all very good.	
# But if we really want to test its effectiveness,	
# we have to ensure that the neural net only	
# sees the training data, and that it never sees the test data.	
# So we have to rewrite our code to ensure that the tokenizer is	
# just fit to the training data.
# Let's take a look at how to do that now.

vocab_size = 10000
max_length = 9
embedding_dim = 16
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
# We'll first instantiate a tokenizer like before,	
# but now, we'll fit the tokenizer on just the training sentences	
# that we split out earlier, instead of the entire corpus.
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

# And now, instead of one overall set of sequences,	
# we can now create a set of training sequences,	
training_sequences = tokenizer.texts_to_sequences(training_sentences)

# and pad them, and then do exactly the same thing	
# for the test sequences.
training_padded = pad_sequences(training_sequences, maxlen=max_length, 
                                padding='post', truncating='post')

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, 
                               padding='post', truncating='post')
import keras
import numpy as np

training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)


model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

num_epochs = 30

history = model.fit(training_padded, training_labels, epochs=num_epochs,
                    validation_data=(testing_padded, testing_labels), verbose=2)

sentence =[
    "granny starting to fear spiders in the garden might be real",
    "the weather today is bright and sunny"
]

sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, 
                       padding='post', truncating='post')

print(model.predict(padded))


