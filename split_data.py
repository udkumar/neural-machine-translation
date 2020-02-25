from pickle import load
from pickle import dump
from numpy.random import rand
from numpy.random import shuffle
#import keras
#from keras.datasets import reuters

'''
Running the example creates three new files:
the english-german-both.pkl that contains all of the train and test examples
that we can use to define the parameters of the problem, such as max phrase lengths and the vocabulary,
and the english-german-train.pkl and english-german-test.pkl
files for the train and test dataset.
'''

# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

# load dataset
raw_dataset = load_clean_sentences('eng-hindi.pkl')

# reduce dataset size
n_sentences = 2770
dataset = raw_dataset[:n_sentences, :]
# random shuffle
shuffle(dataset)
# split into train/test
train, test = dataset[:2700], dataset[2700:]
# save
save_clean_data(dataset, 'english-hindi-both.pkl')
save_clean_data(train, 'english-hindi-train.pkl')
save_clean_data(test, 'english-hindi-test.pkl')

'''
develop the neural translation model.
Let’s start off by loading the datasets so that we can prepare the data.
The function below named load_clean_sentences() can be used to load the train, test, and both datasets in turn.
'''
# load a clean dataset
def load_clean_sentences(filename):
    return load(open(filename, 'rb'))

# load datasets
dataset = load_clean_sentences('english-hindi-both.pkl')
train = load_clean_sentences('english-hindi-train.pkl')
test = load_clean_sentences('english-hindi-test.pkl')

'''
We will use the “both” or combination of the train and test datasets to define the maximum
length and vocabulary of the problem.

We can use the Keras Tokenize class to map words to integers, as needed for modeling.
We will use separate tokenizer for the English sequences and the Hindi sequences.
The function below-named create_tokenizer() will train a tokenizer on a list of phrases.
'''
# fit a tokenizer
def create_tokenizer(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

'''
Similarly, the function named max_length() below will find the length of the
longest sequence in a list of phrases.
'''
# max sentence length
def max_length(lines):
    return max(len(line.split()) for line in lines)

'''
We can call these functions with the combined dataset to prepare tokenizers, vocabulary sizes, and maximum lengths for
both the English and Hindi phrases.
'''

# prepare english tokenizer
eng_tokenizer = create_tokenizer(dataset[:, 0])
eng_vocab_size = len(eng_tokenizer.word_index) + 1
eng_length = max_length(dataset[:, 0])
print('English Vocabulary Size: %d' % eng_vocab_size)
print('English Max Length: %d' % (eng_length))

# prepare Hindi tokenizer
ger_tokenizer = create_tokenizer(dataset[:, 1])
ger_vocab_size = len(ger_tokenizer.word_index) + 1
ger_length = max_length(dataset[:, 1])
print('German Vocabulary Size: %d' % ger_vocab_size)
print('German Max Length: %d' % (ger_length))