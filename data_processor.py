import numpy as np
import csv
import re
import pandas
from string import punctuation
from datetime import datetime
from collections import Counter
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
sentiments = []
text = []
senti = []
dates = []
# Clean the original data,
#Remove punctuations, lower case, special characters
def strip_non_ascii (string):
    stripped = (c for c in string if 0<ord(c)<127)
    return ''.join(stripped)

filename = input ('please input name of the file you want to analyze:')
with open(filename,'r',encoding='utf8') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        current_date = datetime.strptime(row[1],'%m/%d/%Y %H:%M')
        current_date = datetime.strftime(current_date,'%m/%Y')
        dates.append(current_date)
        sentiment=dict()
        sentiment['orig'] = row[2]
        sentiment['dealercode'] = int(row[5])
        sentiment['sentiment'] = row[3]
        sentiment['date'] = current_date
        if re.match(r'^RT.*',sentiment['orig']):
            continue
        sentiment['clean'] = sentiment['orig']
        sentiment['clean'] = strip_non_ascii(sentiment['clean'])
        sentiment['clean'] = sentiment['clean'].lower()
        sentiment['clean'] = re.sub('[^0-9a-zA-Z]+', ' ', sentiment['clean'])
        sentiment['clean'] = re.sub('[0-9]+', ' ', sentiment['clean'])
        sentiment['senti'] = sentiment['sentiment']
        sentiment['senti'] = strip_non_ascii(sentiment['senti'])
        sentiments.append(sentiment)
        text.append(sentiment['clean'])
        #print (text)
        senti.append(sentiment['senti'])

#Split the texts into words
all_text = '|'.join(str(i) for i in text)
discussions = all_text.split('|')
all_text=' '.join(discussions)
words = all_text.split()

# store a dictionary containing key words mapped to indexes
int_words = {}
# store a list of output
output_comments = []

#print (words[:100])
def get_vocab_to_int(words):
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    result = {}
    for i, word in enumerate(vocab, 1):
        # map the index to the word
        result[word] = i
        # map the words to the index
        int_words[i] = word
    return result

def get_review_ints(vocab_to_int, review):
    review_ints = []
    for row in review:
        review_ints.append([vocab_to_int[word] for word in row.split()])
    return review_ints

vocab_to_int = get_vocab_to_int(words)
review_ints = get_review_ints(vocab_to_int, discussions)
#test = get_review_ints(vocab_to_int, ['corolla door speaker wire harness corroded replaced harness good report good photos   tacoma tonneau cover'])
#print(test)
print(len(vocab_to_int))

all_senti = '|'.join (str(i) for i in senti)
senti = np.array([0 if label=='negative' else 1 for label in all_senti.split('|')])
labels = senti

review_lens = Counter([len(x) for x in review_ints])
print('Zero-length reviews:{}'.format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))
#remove the reviews with length = 0 
non_zeros_idx = []
non_zeros_idx = [ii for ii, review in enumerate(review_ints) if len(review) != 0]
#print(non_zeros_idx)
print(len(non_zeros_idx))
review_ints = [ review_ints[ii] for ii in non_zeros_idx]
labels = np.array( [labels[ii] for ii in non_zeros_idx])
print(labels)
#make every review same length (100), when the length <100, add 0 to the left
seq_len = 100
features = np.zeros((len(review_ints), seq_len), dtype=int)
for i,review in enumerate(review_ints):
    features[i, -len(review):] = np.array(review)[:seq_len]
print(features[:10,:100])
#split training, testing and validation set, fraction 1:9
split_frac = 0.9
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
embed_size = 200

#with graph.as_default(): 
      #embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1)) 
      #embed = tf.nn.embedding_lookup(embedding, inputs_) 

with graph.as_default():
    embedding = tf.Variable(tf.truncated_normal((n_words, embed_size), stddev=0.01))
    embed = tf.nn.embedding_lookup(embedding, inputs_)

with graph.as_default():
    lstm = tf.contrib.rnn.LSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, keep_prob)
    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)

with graph.as_default():
    outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
print(outputs)

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

#Training, this is where the problem happens. need to use the big machine for the training process otherwise it's too slow.        
epochs = 10
with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
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
    saver.save(sess, "checkpoints/sentiment.ckpt")
test_acc = []
test_pred = []

# function to convert the list of index back to the original string
def convert_to_words(data, output):
    ''' (review_ints, empty list) -> None
    Take in the list of lists containing the indexes of the words,
    find the corresponding words of the indexes, and store them
    into a list of lists containing strings.
    (ex. [["for example"], ["original string"]])
    '''
    for each_ints in data:
        curr_words = ""
        for each_int in each_ints:
            curr_words += " " + int_words[each_int]
        curr_words = curr_words[1:]
    output.append([curr_words])

# store to output_comments variable
# convert_to_words(review_ints, output_comments)

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
end_idx = start_idx + len(test_pred_flat)+1
sentiment.loc[start_idx:end_idx,'predicted_sentiment'] = test_pred_flat 
print (test_pred_flat)
#from os import path
#from wordcloud import WordCloud
#import matplotlib.pyplot as plt
#start_idx = len(train_x) + len(val_x) 
#test_tweets = Tweet[start_idx:] 
  
#fig = plt.figure( figsize=(40,40)) 
#sub1= fig.add_subplot(2,2,1) 
#posActualTweets = test_tweets[test_tweets.sentiment==1] 
#posPredTweets = test_tweets[test_tweets.predicted_sentiment==1] 
#tweetText = ' '.join((posActualTweets['clean_tweet'])) 
#wordcloud = WordCloud().generate(tweetText) 
#plt.title("Positive Sentiment - Actual") 
#plt.imshow(wordcloud, interpolation='bilinear') 
#plt.axis("off") 
#sub2= fig.add_subplot(2,2,2) 
#plt.title("Positive Sentiment - Prediction") 
#tweetText = ' '.join((posPredTweets['clean_tweet'])) 
#wordcloud = WordCloud().generate(tweetText) 
#plt.imshow(wordcloud, interpolation='bilinear') 
#plt.axis("off") 
#plt.show() 
#negPredTweets = test_tweets[test_tweets.predicted_sentiment!=1] 
#tweetText = ' '.join((negActualTweets['clean_tweet'])) 
#fig = plt.figure( figsize=(20,20)) 
#sub1= fig.add_subplot(2,2,1) 
#wordcloud = WordCloud().generate(tweetText) 
#plt.imshow(wordcloud, interpolation='bilinear') 
#plt.title("Negative Sentiment - Actual") 
#plt.axis("off") 
#sub2= fig.add_subplot(2,2,2) 
#tweetText = ' '.join((negPredTweets['clean_tweet'])) 
#wordcloud = WordCloud().generate(tweetText) 
#plt.title("Negative Sentiment - Prediction") 
#plt.imshow(wordcloud, interpolation='bilinear') 
#plt.axis("off") 
#plt.show() 

