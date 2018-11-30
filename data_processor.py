import numpy as np
import sklearn
import csv
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import re
import pandas
from string import punctuation
from datetime import datetime
import tensorflow as tf
import os
from os import path
from wordcloud import WordCloud
from collections import Counter
nltk.download('stopwords')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

filename = input ('please input name of the file you want to analyze:')
comment_table, words = pandas.read_csv(filename), []
custom_stop_words,stops = ["dealer", "dealers"],set(stopwords.words("english"))

# filter the comments
def comment_to_words(comments, words):
   '''(str) -> str
   Clean up the comment and keep only meaningful words'''
   # remove special characters
   terms = re.sub("[^0-9a-zA-Z&.,%]", " ", comments)
   meaningful_terms = terms.lower().split()
   clean_comment(meaningful_terms)
   words += meaningful_terms
   return " ".join(meaningful_terms)

def clean_comment(terms):
   '''(list of str) -> None
   Filter each string in the list of comments'''
   i = 0
   while i < len(terms):
      curr = terms[i]
      # remove invalid terms
      if ((not bool(\
         re.match("((^\d+\.?\d+%$)|(^[0-9]*[a-zA-Z]+&?[a-zA-Z0-9]*\.?$))",curr)))\
          or (curr in custom_stop_words) or (curr in stops)):
         terms.pop(i)
         i = i - 1
      # remove unfiltered periods
      elif (curr[-1] == "." or curr[-1] == ","):
         terms[i] = curr[:-1]
      # convert percentage into float then round down
      elif (curr[-1] == "%"):
         terms[i] = str(int(float(curr[:-1])/10))
      i += 1
   return None

# convert each sentiment into corresponding integer
def read_sent(sentiment):
   '''(str) -> int
   Convert negative to 0, positive to 1, neutral to 2'''
   result = 2
   if sentiment == "negative":
      result = 0
   elif sentiment == "positive":
      result = 1
   return result

# store cleaned comment and converted sentiment into the comment table
comment_table['cleaned'] =comment_table['DISCUSSION_POINTS__C'].apply(lambda x: comment_to_words(x, words))
# negative is 0, positive is 1, neutral is 2
comment_table['senti'] = comment_table['Sentiment'].apply(lambda x: read_sent(x))

# generate the vocabulary list from the whole text
counts = Counter(words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = { word : i for i, word in enumerate(vocab, 1)}

# generate the comment data represented with integers
comment_ints = []
for each in comment_table['cleaned']:
   comment_ints.append([vocab_to_int[word] for word in each.split()])

labels = np.array([read_sent(each) for each in comment_table['Sentiment'][:]])

comment_lens = Counter([len(x) for x in comment_ints])
print('Zero-length reviews:{}'.format(comment_lens[0]))
print("Maximum comment length: {}".format(max(comment_lens)))

# comment_idx = [idx for idx, comment in enumerate(comment_ints) if len(comment) >0]
# labels = labels[comment_idx]
# comment_table = comment.ix[comment_idx]
# comment_ints = [comment for comment in comment_ints if len(comment)>0]

#
seq_len =100
features = np.zeros((len(comment_ints), seq_len), dtype=int)
for i,row in enumerate(comment_ints):
    features[i, -len(row):] = np.array(row)[:seq_len]
print(features[:10,:100])
split_frac = 0.8
split_index = int(len(features)*split_frac)
train_x, val_x = features[:split_index], features[split_index:]
train_y, val_y = labels[:split_index], labels[split_index:]

test_index = int(len(val_x)*0.5)
val_x, test_x = val_x[:test_index], val_x[test_index:]
val_y, test_y = val_y[:test_index], val_y[test_index:]

print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape), 
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

#lstm_size: Number of units in the hidden layers in the LSTM cells. Usually larger is better performance wise. Common values are 128, 256, 512, etc.
lstm_size = 256
#lstm_layers: Number of LSTM layers in the network. I'd start with 1, then add more if I'm underfitting.
lstm_layers = 1
#batch_size: The number of reviews to feed the network in one training pass. Typically this should be set as high as you can go without running out of memory.
batch_size = 512
learning_rate = 0.001

n_words = len(vocab_to_int)
# Create the graph object
graph = tf.Graph()
# Add nodes to the graph
with graph.as_default():
    inputs_ = tf.placeholder(tf.int32, [batch_size, seq_len], name='inputs')
    labels_ = tf.placeholder(tf.int32, [None, None], name='labels')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
# Size of the embedding vectors (number of units in the embedding layer)
embed_size = 300

with graph.as_default(): 
      embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1)) 
      embed = tf.nn.embedding_lookup(embedding, inputs_) 

#with graph.as_default():
 #   embedding = tf.Variable(tf.truncated_normal((n_words, embed_size), stddev=0.01))
  #  embed = tf.nn.embedding_lookup(embedding, inputs_)

with graph.as_default():
    lstm = tf.contrib.rnn.LSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)

with graph.as_default():
    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, activation_fn=tf.sigmoid)
    cost = tf.losses.mean_squared_error(labels_, predictions)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

with graph.as_default():
    correct_pred = tf.equal( tf.cast(tf.round(predictions), tf.int32), labels_ )
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#batch size should be bigger, it's only 28 becuase that the sample size is too small
def get_batches(x, y, batch_size=100):
    n_batches = len(x) // batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
print(comment_table)
print(len(train_x)+len(val_x))
print(len(train_x)+len(val_x)+512-1)
epochs = 10
with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1

    for e in range(epochs):
        state = sess.run(initial_state)
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_ : x,
                    labels_ : y[:, None],
                    keep_prob : 0.5,
                    initial_state : state}

            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            if iteration % 5 == 0:
                print('Epoch: {}/{}'.format(e, epochs),
                      'Iteration: {}'.format(iteration),
                      'Train loss: {}'.format(loss))
            if iteration % 25 == 0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))

                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_ : x,
                            labels_ : y[:, None],                            
                            keep_prob : 1,
                            initial_state : val_state}

                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print('Val acc: {:.3f}'.format(np.mean(val_acc)))
            iteration += 1
test_acc = []
test_pred = []
with tf.Session(graph=graph) as sess:
    saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
    test_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for ii, (x, y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
        feed = {inputs_: x,
                labels_: y[:, None],
                keep_prob: 1,
                initial_state: test_state}
        batch_acc, test_state = sess.run([accuracy, final_state], feed_dict=feed)
        test_acc.append(batch_acc)
        prediction = tf.cast(tf.round(predictions),tf.int32)
        prediction = sess.run(prediction,feed_dict=feed)
        test_pred.append(prediction)
    print("Test accuracy: {:.3f}".format(np.mean(test_acc)))
    print(test_pred)
test_pred_flat = (np.array(test_pred)).flatten()
start_idx = len(train_x) + len(val_x)
print(len(test_pred_flat))
end_idx = start_idx + len(test_pred_flat)-1
comment_table.loc[start_idx:end_idx,'predicted_sentiment'] = test_pred_flat 
comment_table.to_csv('predictions.csv')
