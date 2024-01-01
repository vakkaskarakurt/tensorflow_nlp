import tensorflow as tf 
from tensorflow import keras
from keras.preprocessing.text import Tokenizer 
#First of all, we need the tokenize our API's 
# and we can get these from Tensorflow Keras like this.

# We can represent our sentences as a Python array	
# of strings like this.
sentences = [ 
    "I love my dog",
    "I love my cat",
    ]

# The num_words parameter is the maximum number	
# of words to keep.	
# So instead of, for example, just these two sentences,	
# imagine if we had hundreds of books to tokenize,	
# but we just want the most frequent	
# 100 words in all of that.
tokenizer = Tokenizer(num_words=100) 
# when we do the next step, and that's	
# to tell the tokenizer to go through all the text	
# and then fit itself to them like this.
tokenizer.fit_on_texts(sentences)
# The full list of words is available as the tokenizer's	
# word index property.
word_index = tokenizer.word_index

print("Word Index: ", word_index)


# So for example, if we updated our sentences to this
# by adding a third sentence, noting that "dog" here	
# is followed by an exclamation mark,	
# the nice thing is that the tokenizer	
# is smart enough to spot this and not create a new token.	
# It's just "dog."	
# And you can see the results here.	
# There's no token for "dog exclamation,"	
# but there is one for "dog."

sentences = [
    "I love my dog",
    "I love my cat",
    "You love my dog!"
    ]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("Word Index 2: ", word_index)