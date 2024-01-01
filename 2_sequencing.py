from keras.preprocessing.text import Tokenizer

# We'll add another sentence to our set of texts,	
# and I'm doing this because the existing sentences all	
# have four words, and it's important to see	
# how to manage sentences, or sequences,	
# of different lengths.
sentences = ["I love my dog",
             "I love my cat",
             "You love my dog!",
             "Do you think my dog is amazing?"]



tokenizer=Tokenizer(num_words=100)

# Let's now look back at the code.	
# I have a set of sentences that I'll use	
# for training a neural network.	
tokenizer.fit_on_texts(sentences)


word_index=tokenizer.word_index


# The tokenizer supports a method called texts	
# to sequences which performs most of the work for you.	
# It creates sequences of tokens representing each sentence.

# The tokenizer gets the word index from these	
# and create sequences for me.
sequences=tokenizer.texts_to_sequences(sentences)

print("Word Index: ", word_index)
print("Sequences: ", sequences)

# So now, if I want to sequence these sentences, containing	
# words like manatee that aren't present in the word index,	
# because they weren't in my initial set of data,	
# what's going to happen?
test_data = ["I really love my dog",
             "my dog loves my manatee"
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("Test Sequences: ", test_seq)
# We see this, I really love my dog.	
# A five-word sentence ends up as 4, 2, 1, 3,	
# a four-word sequence.	
# Why?	
# Because the word really wasn't in the word index.	
# The corpus used to build it didn't contain that word.	
# And my dog loves my manatee ends up	
# as 1, 3, 1, which is my, dog, my,	
# because loves and manatee aren't in the word index.	
# So as you can imagine, you'll need a really big word index	
# to handle sentences that are not in the training set.	
# But in order not to lose the length of the sequence,	
# there is also a little trick that you can use.
# By using the OOV token property, and setting it as something	
# that you would not expect to see in the corpus, like angle	
# bracket, OOV, angle bracket, the tokenizer	
# will create a token for that, and then	
# replace words that it does not recognize	
# with the Out Of Vocabulary token instead.

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

test_seq = tokenizer.texts_to_sequences(test_data)
print("Word Index with <OOV>: ", word_index)
print("Test Sequences with <OOV>: ", test_seq)



# And while it helps maintain the sequence length	
# to be the same length as the sentence,	
# you might wonder, when it comes to needing	
# to train a neural network, how it can handle	
# sentences of different lengths?	
# With images, they're all usually the same size.	
# So how would we solve that problem?	
# The advanced answer is to use something	
# called a RaggedTensor.	
# That's a little bit beyond the scope of this series,	
# so we'll look at a different and simpler solution, padding.
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

sentences = ["I love my dog",
                "I love my cat",
                "You love my dog!",
                "Do you think my dog is amazing?"]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences)
print("Word Index with <OOV>: ", word_index)
print("Sequences with <OOV>: ", sequences)
print("Padded Sequences: ", padded)

padded = pad_sequences(sequences, padding='post')
print("Padded Sequences with post: ", padded)
padded = pad_sequences(sequences, padding='post', maxlen=5)
print("Padded Sequences with post and maxlen=5: ", padded)
padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=5)
print("Padded Sequences with post, truncating=post, and maxlen=5: ", padded)

