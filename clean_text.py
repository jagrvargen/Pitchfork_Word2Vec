'''
   Create tokenized word list
'''
import collections
import datetime as dt
import json
import nltk
import numpy as np
import math
import random
import re
import tensorflow as tf
from lstm import *
from nltk.tokenize import wordpunct_tokenize


def split_text(file_in):
    '''Convert text file into list of all words'''
    text = list()
    extras = set(['\\\\', r'"', r'-', '--', '(', ')', '.\\', ',\\', r'."', r'".', ',"', '",', "''", '.\\\\', r'.\\', r',\\', '!', ',', '``', ';', ':', r"'", '\\\\\\\\n', r"'\\n", r'\\r\\n', '/', '&', '!")', '.,"', r'," "', r'" "', r'"),', ")))'", '..."'])

    with open(file_in, 'r', encoding='utf-8') as f:
        try:
            for line in f:
                line = re.sub('[^A-Za-z\s.]+', '', line, re.I)
                line = line.lower()
                line = wordpunct_tokenize(line)
                for word in line:
                    if word not in extras and not re.search('[a-z]+[0-9]+', word):
                        text.append(word)
        except Exception as e:
            print(str(e))
            
    return text

def build_dataset(text, vocab_size):
    '''
       Create mappings of data to their counts and indexes.

       Dictionary: n most common words including the 'UNK' token mapped to
       their integer id value in the dataset.

       Indexes: A dictionary mapping ids to vocabulary words in order to
       translate tensors back into words.

       Data: A translation of the whole text into integer values.

       Count: A list of n most common word tokens paired with their total
       number of occurences in text.
    '''
    count = [['<unk>', 0]]
    count.extend(collections.Counter(text).most_common(vocab_size - 1))
    dictionary = dict()
    indexes = dict()
    for i in range(1, len(count)):
        dictionary[count[i][0]] = i
        indexes[i] = count[i][0]

    dictionary['<unk>'] = 0
    indexes[0] = '<unk>'
    data = list()
    for word in text:
        if word not in dictionary:
            count[0][1] += 1
        else:
            data.append(dictionary[word])

    return data, count, dictionary, indexes

def batchify(data, context_window, sample_number, batch_size, data_index=0):
    '''
       Splits data into batches of size batch_size, which are ndarrays
       containing words with some associated context words drawn from
       the context_window on either side. sample_number is the
       number of context words to pull before shifting the context_window.
       data_index indicates where in the dataset to begin the batching
       process and defaults to 0.
    '''
    window = context_window * 2 + 1
    inputs = np.ndarray(shape=(batch_size), dtype=np.int32)
    context = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    for i in range(batch_size // sample_number):
        index = i + data_index + (window // 2 + 1)
        input = data[index]
        c_range = [r for r in range(index - context_window, index +
                                    context_window + 1) if r != index]
        for j in range(sample_number):
            inputs[i * sample_number + j] = input
            context[i * sample_number + j] = data[random.choice(c_range)]

    data_index += 1

    return inputs, context, data_index

def init_embed(batch_size, count, dictionary, embedding_size, num_sampled, sample_size, vocab_size):
    '''
       Set up the embedding tensors with 20 randomly selected training examples
       from the most common 100 words from the dataset.
    '''
    # Select 20 of 100 most common words by id and place in a tensor.
    common_words = [count[i][0] for i in range(1, num_sampled + 1)]
    common_words = [dictionary[word] for word in common_words]
    cv_samples = np.random.choice(common_words, size=sample_size, replace=False)
    cv_data = tf.constant(cv_samples, dtype=tf.int32)

    # Create placeholder tensors for input and context training values.
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_context = tf.placeholder(tf.int32, shape=[batch_size, 1])

    # Create a vocab_size x embedding_size matrix of randomly distributed
    # values between -1 and 1. embed_lookup acts as indexing reference
    # for inputs
    embeddings = tf.Variable(tf.random_uniform([vocab_size,
                                                embedding_size],
                                               -1.0, 1.0))
    embed_lookup = tf.nn.embedding_lookup(embeddings, train_inputs)

    return train_inputs, train_context, cv_data, cv_samples, embeddings, embed_lookup

def optimize_nce(vocab_size, embedding_size, embed_lookup, train_context, num_sampled):
    # Initialize weights and biases for training
    weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size],
                                              stddev=1.0 /
                                              math.sqrt(embedding_size)))
    biases = tf.Variable(tf.zeros([vocab_size]))

    # Train weights and biases by checking Noise Contrastive Estimation
    # loss.
    loss = tf.nn.nce_loss(
        weights=weights,
        biases=biases,
        inputs=embed_lookup,
        labels=train_context,
        num_sampled=num_sampled,
        num_classes=vocab_size)
        
    # Adam Optimizer
    cost = tf.reduce_mean(loss)
    op = tf.train.AdamOptimizer().minimize(cost)
    
    return cost, op

def similarity_check(embeddings, cv_data):
        '''Cosine similarity'''
        normalize = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        norm_embed = embeddings / normalize

        cv_embed = tf.nn.embedding_lookup(norm_embed, cv_data)
        similarity = tf.matmul(cv_embed, norm_embed, transpose_b=True)

        return similarity, norm_embed

def train(graph, steps, file_in):
    avg_loss = 0
    data_index = 0
    vocab_size = 50000
    embedding_size = 300
    batch_size = 128
    sample_number = 2
    sample_size = 20
    num_sampled = 100
    context_window = 2

    text = split_text(file_in)

    data, count, dictionary, indexes = build_dataset(text, vocab_size)

    with graph.as_default():
        train_inputs, train_context, cv_data, cv_samples, embeddings, embed_lookup = init_embed(batch_size, count, dictionary, embedding_size, num_sampled, sample_size, vocab_size)

        cost, op = optimize_nce(vocab_size, embedding_size, embed_lookup, train_context, num_sampled)

        similarity, norm_embed = similarity_check(embeddings, cv_data)

        init = tf.global_variables_initializer()
    """
    saver = tf.train.Saver()
    """
    with tf.Session(graph=graph) as sess:

        init.run()

        for step in range(steps):
            inputs, context, data_index = batchify(data, context_window,
                                                   sample_number, batch_size,
                                                   data_index)
            feed_dict = {train_inputs: inputs, train_context: context}

            _, loss = sess.run([op, cost], feed_dict=feed_dict)
            avg_loss += loss

            if step % 1000 == 0:
                if step > 0:
                    avg_loss /= 1000
                print("Step is {}".format(step))
                print('Average loss is {}'.format(avg_loss))
                print('Loss is {}'.format(loss))
                avg_loss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(20):
                    valid_word = indexes[cv_samples[i]]
                    neighbors = 8
                    nearest = (-sim[i, :]).argsort()[1:neighbors + 1]
                    output = "Closest words to {} are".format(valid_word)
                    for k in range(neighbors):
                        close_word = indexes[nearest[k]]
                        output = "{} {}".format(output, close_word)
                    print(output)

        return text, dictionary, norm_embed

if __name__ == "__main__":
    graph = tf.Graph()

    num_steps = 50000
    text, dictionary, norm_embed = train(graph, num_steps, 'reviews_corpus.txt')
