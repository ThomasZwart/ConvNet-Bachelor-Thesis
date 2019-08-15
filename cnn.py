import tensorflow as tf
import numpy as np
import time

tf.reset_default_graph() 

# De data wordt opgehaald en opgeslagen in numpy array
data = (np.load("testdata_21_classes_1000.npy")).copy()
# De data wordt gerandomiseerd
np.random.shuffle(data)

# Splits de data op in train en validatie data (test data in test fase)
train_data = data[:800000]
test_data = data[800000:820000]

# Maak test data
test_x, test_y = [], []
for datapoint in test_data:
    test_x.append(datapoint[0])
    test_y.append(datapoint[1]) 
test_x, test_y = np.array(test_x), np.array(test_y)

# Object voor de training data
class Dataset(object):
    def __init__(self, data):
        self.index = 0
        self.length = len(data)
        
        self.train_x = []
        self.train_y = []
        for datapoint in data:
            self.train_x.append(datapoint[0])
            self.train_y.append(datapoint[1])            
        self.train_x, self.train_y = np.array(self.train_x), np.array(self.train_y)
        self.shuffle()

    # Returned een nieuwe batch data
    def next_batch(self, batch_size):
        start = self.index
        self.index += batch_size
        end = self.index
        return self.train_x[start:end], self.train_y[start:end]

    # Randomiseert de data
    def shuffle(self):
        p = np.random.permutation(len(self.train_x))
        self.train_x = self.train_x[p].copy()
        self.train_y = self.train_y[p].copy()
        

train_data = Dataset(train_data)

# Set de parameters
use_saved_model = True
n_classes = 21
batch_size = 256
index_epoch = 0
# 1 epoch = eenmaal vooruit en eenmaal terug door het hele netwerk met alle data
# Na 300 epochs stopt het model met trainen, 
# daarna kan altijd nog verder worden getrained aangezien het model wordt opgeslagen
hm_epochs = 300
# Tegen overfitting, dropout = 1 - keep_rate
keep_rate_conv1 = 1


x = tf.placeholder('float', [None, 784])
# Labels
y = tf.placeholder('float')

# Convolutie algoritme
def conv2d(x, W):
    # Strides is hoeveel de convolutionele window beweegt, grootte window is bepaald in parameter W
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = "SAME")

# Maxpool algoritme
def maxpool2d(x):
    # ksize = window size in form [batchnmr, height, width, channels], channels is 1 here, only gray in Mnist
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides = [1,2,2,1], padding = "SAME")

# Maakt het neurale netwerk
def convolutional_neural_network(x):
    # CNN krijgt een 28 bij 28 foto, er gaat een 5 bij 5 filter over de foto die random geïnitialiseerd is
    
            # 5 bij 5 convolutie, 1 input (the picture), 32 features/outputs
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               # 5 bij 5 convolutie, 32 inputs (previous layer), 64 outputs
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               # Een 7 bij 7 foto, want we hebben 2x gepooled met stride 2, dat geeft ons 28/2/2 = 
               # Dus de volledig verbonden laag (fc) krijgt 64 geconvoleerde foto's van 7 bij 7 pixels
               'W_fc':tf.Variable(tf.random_normal([7*7*64,1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    # Biases voor de gewichten
    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Relu is f(x) = max(0, x)
    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)
    conv1 = tf.nn.dropout(conv1, keep_rate_conv1)

    conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc = tf.reshape(conv2,[-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

    output = tf.matmul(fc, weights['out']) + biases['out']
    return output

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    # Initialiseert de loss functie en het optimaliseer algoritme
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = prediction) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Start een tensorflow sessie
    with tf.Session() as sess:
        saver = tf.train.Saver() 

        # Train een nieuw model (of een oud model verder trainen)
        if not use_saved_model:
            sess.run(tf.global_variables_initializer())
            
            # Als je een oud model wilt ophalen om verder te trainen, gebruik deze code:
            #saver.restore(sess, "./model/batch_size_256_dropout_05_conv1.ckpt")

            for epoch in range(hm_epochs):
                # Initialisatie elke epoch
                epoch_loss = 0
                train_data.index = 0
                # Shuffle data na elke epoch, zodat de (nieuwe) batches in een andere volgorde door het netwerk gaan
                train_data.shuffle()
                start = time.time()
                # Elke iteratie is 1x de batch size, wanneer deze klaar is, zijn alle datapunten door het netwerk gesepareerd in batches
                for _ in range(int(train_data.length/batch_size)):
                    # Training data en labels
                    epoch_x, epoch_y = train_data.next_batch(batch_size)
                    # Draai de sessie met de data, optimizer en kosten functie
                    _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                    epoch_loss += c
                                   
                print('Epoch', epoch + 1, 'completed out of', hm_epochs,'loss:', epoch_loss)
                print('Time', time.time() - start)

                # Evalueer het netwerk
                correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))      

                # Sla het netwerk op om de 10 epochs
                if (epoch % 10 == 0):
                    saver.save(sess, "./model/batch_size_256_dropout_05_conv1.ckpt")
                  
            saver.save(sess, "./model/batch_size_256_dropout_05_conv1.ckpt")

        # Evalueer een oud model
        else:
            saver.restore(sess, "./model/batch_size_256_dropout_00.ckpt")
            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy:', accuracy.eval({x:test_x, y:test_y}))
            
train_neural_network(x)

