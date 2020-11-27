#!/usr/bin/env python
# coding: utf-8

# ## Text Generation

# Using deep learning model to generate Harry Potter-liked stories . Given a sequence of words as seed, this model predicts the following words.

# #### Import required python packages

# In[1]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import initializers
import string
import time
from random import randint
import numpy as np
from pickle import dump, load
from gensim.models import Word2Vec
import re
import os
import tensorflow as tf

print(tf.__version__)


# #### Prepare the training datasets

# In[2]:


# Prepare dataset
directory = 'HarryPotter-en/'
fnames = os.listdir(directory)
fnames.sort()

txt_data_files = []
for f in fnames:
    txt_data_files.append(os.path.join(directory, f))
txt_data_files


# In[3]:


# load text data
def load_txt_data(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


# In[4]:


# load dataset
txt_data_set = ''
for f in txt_data_files:
    txt_data_set += load_txt_data(f)
    txt_data_set = load_txt_data(f)
print(txt_data_set[:200])


# #### Pre-process the text data

# 1. replace '--' with a space ' '.
# 2. tokenize the text by whits space
# 3. remove puntuation from each token
# 4. remove all tokens that are not alphabetic
# 5. convert to lower case

# In[5]:


def preprocess_txt_dataset(txt_data):
    txt_data = txt_data.replace('--', ' ')  # replace '--' with a space ' '
    txt_data = txt_data.lower().strip()
    # creating a space between a word and the punctuation following it    
    #txt_data = re.sub(r"([?.!,¿])", r" \1 ", txt_data)
    #txt_data = re.sub(r'[" "]+', " ", txt_data)
    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    #txt_data = re.sub(r"[^a-zA-Z?.!,¿]+", " ", txt_data)
    txt_data = txt_data.rstrip().strip()
    
    tokens = txt_data.split()  # split into tokens by white space
    table = str.maketrans('', '', string.punctuation)  # remove punctuation from each token
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]  # remove remaining tokens that are not alphabetic
    #tokens = [word.lower() for word in tokens]  # make lower case
    return tokens


# In[6]:


# Pre-process and tokenize document
tokens = preprocess_txt_dataset(txt_data_set)
print(tokens[:200])
print('Total Tokens: %d' % len(tokens))
print('Unique Tokens: %d' % len(set(tokens)))


# Organize the text data into sequences of tokens. Each sequence has a length of 50+1 tokens.

# In[7]:


# organize into sequences of tokens
length = 50 + 1
#length = 7 + 1
txt_sequences = list()

for i in range(length, len(tokens)):
    # select sequence of tokens
    seq = tokens[i-length:i]
    # convert into a line
    line = ' '.join(seq)
    # store
    txt_sequences.append(line)
print('Total Sequences: %d' % len(txt_sequences))


# In[8]:


def save_txt_data(txt_lines, filename):
    text = '\n'.join(txt_lines)
    file = open(filename, 'w')
    file.write(text)
    file.close()


# In[9]:


# save sequences to file
out_filename = 'HarryPotter_series.txt'
save_txt_data(txt_sequences, out_filename)


# #### Tokenize

# Tokenize the text sequences by Tokenizer of keras. The output is sequences of word index.

# In[10]:


# integer encode sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(txt_sequences)
sequences = tokenizer.texts_to_sequences(txt_sequences)


# There are total 7731 vocablaries.

# In[11]:


# vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)


# Prepare word sequences for gensim word2vec model training. The word_sequences is a 2D array of tokens.

# In[12]:


word_sequences = []

for i in range(len(txt_sequences)):
    word_sequences.append(txt_sequences[i].split())


# In[13]:


print(word_sequences[0:2])


# In[14]:


embedding_dim = 256  # embedding dimension of word2vec model


# #### gensim model training

# In[15]:


gensim_model = Word2Vec(word_sequences, size = embedding_dim, iter=5, sg=1, window=5, min_count=5, max_vocab_size=None, hs=0, negative=5, workers=3)


# In[16]:


gensim_model.save('word2vec.model')


# In[17]:


gensim_model = Word2Vec.load('word2vec.model')


# Read the word vectors trained by gensim and prepare an embedding matrix for the initialization of Embedding layer.

# In[18]:


# Prepare embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))

for w in gensim_model.wv.vocab:
    word_vectors = np.array([gensim_model[w]])
    embedding_matrix[tokenizer.word_index[w]] = word_vectors


# Test the performance of the model trained by gensim.

# In[19]:


gensim_model.most_similar('one')


# In[20]:


gensim_model.similarity('harry', 'wizard')


# Prepare the training dataset and labels

# In[21]:


# separate into training data and label
sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)  # One-hot encoding
seq_length = X.shape[1]


# In[22]:


seq_length


# Set the hyper-parameters

# In[23]:


memory_cell = 256
dropout_rate = 0.2


#  #### Define a deep learning model

# Define a deep learning sequential model with one embedding layer initialized by the word vectors pre-trained by gensim, two LSTM layers with 256 memory cells, one dropout layer with dropout rate 0.2, one dense layer with 256 neurons and relu activation, another dropout layer with dropout rate 0.2, and one output dense layer with 7731 neurons and softmax activation.

# In[24]:


# define model
harry_potter_model = Sequential()
harry_potter_model.add(Embedding(vocab_size, embedding_dim, input_length=seq_length, 
    embeddings_initializer=initializers.Constant(embedding_matrix), trainable=False,))
harry_potter_model.add(LSTM(memory_cell, return_sequences=True))
harry_potter_model.add(LSTM(memory_cell))
harry_potter_model.add(Dropout(dropout_rate))
harry_potter_model.add(Dense(memory_cell, activation='relu'))
harry_potter_model.add(Dropout(dropout_rate))
harry_potter_model.add(Dense(vocab_size, activation='softmax'))
harry_potter_model.summary()


# In[25]:


# compile model
harry_potter_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Set the hyper-parameters.

# In[26]:


batch_size = 128
epochs = 400


# #### Train the Harry-Potter style text generator model

# In[27]:


tick1 = time.time()
localtime = time.asctime(time.localtime(time.time()))
print("開始時間是：", localtime)

# fit model
harry_potter_model.fit(X, y, validation_split=0.2, batch_size=batch_size, epochs=epochs)

localtime = time.asctime(time.localtime(time.time()))
print("結束時間是：", localtime)

tick2 = time.time()
print("Elapsed time: ", tick2 - tick1)
print(time.strftime("%H:%M:%S", time.localtime(tick2-tick1-3600*8)))


# In[28]:


# save the model to file
harry_potter_model.save('harry_potter_model.h5')
# save the tokenizer
dump(tokenizer, open('harry_potter_tokenizer.pkl', 'wb'))


# In[29]:


seq_length = len(txt_sequences[0].split()) - 1
print(seq_length)


# In[30]:


# load the model
harry_potter_model = load_model('harry_potter_model.h5')
# load the tokenizer
tokenizer = load(open('harry_potter_tokenizer.pkl', 'rb'))


# #### Stochastic sampling

# Stochastic sampling introduces randomness in the sampling process, by sampling from
# the probability distribution for the next word. In order to control the amount of stochasticity in the sampling process, a parameter called the softmax temperature that characterizes the entropy of the probability distribution is introduced. Given a temperature value, a new probability distribution is computed from the predicted probability distribution of the model. Higher temperature results in higher entropy of the new probability distribution and generates more creative text data. On the contrary, lower temperature results in less randomness of the distribution and generates much more predictable data.

# In[31]:


def txt_gen_sample(preds, temperature=0.5):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[32]:


# generate a sequence from a language model
def generate_txt_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    
    # generate n_words of words
    for _ in range(n_words):
        # tokenize the seed text sequence
        encoded = tokenizer.texts_to_sequences([seed_text])[0]

        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre', truncating='pre')
        # predict probabilities of the next word
        preds = model.predict(encoded, verbose=0)
        # Stochastic sampling with temperature 0.5
        idx = txt_gen_sample(preds[0], 0.5)
        
        # map the predicted index to word
        out_word = tokenizer.index_word[idx]
        
        # append to the seed text
        seed_text += ' ' + out_word
        # append to the generated text sequence
        result.append(out_word)

    return ' '.join(result)


# Prepare the seed text.

# In[33]:


directory = 'generated_story/'

if os.path.isfile('generated_story/gen_story1.txt'):
    os.remove('generated_story/gen_story1.txt')
if os.path.isfile('generated_story/gen_story2.txt'):
    os.remove('generated_story/gen_story2.txt')
if os.path.isfile('generated_story/gen_story3.txt'):
    os.remove('generated_story/gen_story3.txt')
    
fnames = os.listdir(directory)
fnames.sort()
seed_txt_files = []
for f in fnames:
    seed_txt_files.append(os.path.join(directory, f))
seed_txt_files


# Generate 250 words for each seed text.

# In[34]:


num_words = 250

for i, s in enumerate(seed_txt_files):
    seed_text = load_txt_data(s)
    #generate new text
    generated_seq = generate_txt_seq(harry_potter_model, tokenizer, seq_length, seed_text, num_words)
    
    # Save to file
    file = open(directory+'gen_'+fnames[i], 'w')
    file.write(seed_text+generated_seq)
    file.close()
    print(f'Story: {i+1}')
    print(f'Seed Text: {seed_text}')
    print(f'Generated Story: {seed_text+generated_seq}\n')

