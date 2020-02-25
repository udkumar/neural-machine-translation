import string
import re
from pickle import dump
from unicodedata import normalize
from numpy import array
'''
1. Clean Text
First, we must load the data in a way that preserves the Unicode German characters.
The function below called load_doc() will load the file as a blob of text.
# load doc into memory
'''
def load_doc(filename):
    # open the file as read only
    file = open(filename, mode='rt', encoding='utf-8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text
#Each line contains a single pair of phrases, first English and then German, separated by a tab character.

'''
We must split the loaded text by line and then by phrase. The function to_pairs() below will split the loaded text.
# split a loaded document into sentences
'''
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in  lines]
    return pairs
'''
We are now ready to clean each sentence. The specific cleaning operations we will perform are as follows:

Remove all non-printable characters.
Remove all punctuation characters.
Normalize all Unicode characters to ASCII (e.g. Latin characters).
Normalize the case to lowercase.
Remove any remaining tokens that are not alphabetic.
We will perform these operations on each phrase for each pair in the loaded dataset.

The clean_pairs() function below implements these operations.
# Clean a list of lines
'''
def clean_pairs(lines):
    cleaned = list()
    # prepare regex for char filtering
    #re_print = re.compile('[^%s]' % re.escape(string.printable))
    re_print = re.compile(r'[-()\"#/@;:<>{}`+=~|.!?,।]')
    
    # prepare translation table for removing punctuation
    # We can create an empty mapping table, but the third argument of this function
    #allows us to list all of the characters to remove during the translation process.
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        #print(lines)
        clean_pair = list()
        for line in pair:
            # normalize unicode characters
            line = normalize('NFD', line).encode('utf-8', 'ignore')
            #print(line)
            line = line.decode('UTF-8')
            
            # tokenize on white space
            line = line.split()
            #print(line)
            # convert to lowercase
            line = [word.lower() for word in line]
            #print(line)
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            #print(line)
            # remove non-printable chars form each token
            line = [re_print.sub('', w) for w in line]
            #print(line)
            # remove tokens with numbers in them
            line = [word for word in line]
            #print(line)
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return array(cleaned)

'''
Finally, now that the data has been cleaned, we can save the list of phrase pairs to a file ready for use.
The function save_clean_data() uses the pickle API to save the list of clean text to file.
# save a list of clean sentences to file
'''
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

# load dataset
filename = '/home/uday/WorkStation/my_tensorflow/hin-eng/hin.txt'
#filename = '/home/uday/WorkStation/my_tensorflow/deu.txt'

doc = load_doc(filename)
#print(doc)

# split into english-german pairs
pairs = to_pairs(doc)

# clean sentences
clean_pairs = clean_pairs(pairs)
#print(clean_pairs)

# save clean pairs to file
#save_clean_data(clean_pairs, 'german-hindi.pkl')
save_clean_data(clean_pairs, 'eng-hindi.pkl')

#spot check
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i,0], clean_pairs[i,1]))