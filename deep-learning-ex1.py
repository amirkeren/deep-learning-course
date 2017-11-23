
# coding: utf-8

# # Deep Learning - Exercise 1

# ## Covertype Data Set
# 
# Data set that was chosen for this exercise is the Covertype Data Set and the problem is a multiclass classification problem - predicting forest cover type from cartographic variables only (no remotely sensed data).
# 
# [Link to dataset in UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Covertype)

# ### Imports

# In[1]:


import pandas as pd
import tensorflow as tf
import numpy as np

import datetime
import argparse
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer

from helper import clear_model, print_stats, get_batch


# ### Read the data

# In[2]:


df = pd.read_csv('covtype.data', 
                 names = ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 
                          'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 
                          'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 
                          'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 
                          'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 
                          'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 
                          'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 
                          'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 
                          'Soil_Type15', 'Soil_Type16', 'Soil_Type17', 'Soil_Type18', 
                          'Soil_Type19', 'Soil_Type20', 'Soil_Type21', 'Soil_Type22', 
                          'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26', 
                          'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 
                          'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 
                          'Soil_Type35', 'Soil_Type36', 'Soil_Type37', 'Soil_Type38', 
                          'Soil_Type39', 'Soil_Type40', 'Cover_Type'])

X = df.loc[:, df.columns != 'Cover_Type'].astype(float)
y = df.loc[:, 'Cover_Type']


# ### Prepare the data as input for the network

# In[3]:


# Normalize the data
X_values = X.iloc[:, :10].values.reshape(-1, 10)
X.iloc[:, :10] = MinMaxScaler().fit_transform(X_values)
# One hot encode labels
y = pd.DataFrame(MultiLabelBinarizer().fit_transform(y.values.reshape(-1, 1)).astype(float))


# ### Split the data into train, test and validation

# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train, X_validate, y_train, y_validate = train_test_split(X_train, y_train, 
                                                            test_size=0.1, random_state=42)

del X,y


# ### Hyperparameter tuning

# In[ ]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, nargs='?', help='learning rate', default=0.1)
    parser.add_argument("--kp", type=float, nargs='?', help='keep probability', default=0.75)
    parser.add_argument("--rt", type=str, nargs='?', help='regularizer_type', default='L2')    
    parser.add_argument("--rn", type=float, nargs='?', help='regularization', default=0.01)    
    parser.add_argument("--mn", type=int, nargs='?', help='max norm', default=3)
    args = parser.parse_args()

    LR = args.lr
    KP = args.kp
    RT = args.rt
    RN = args.rn
    MN = args.mn 

logging.basicConfig(filename='output.log', level=logging.INFO)


# In[5]:


epochs = 50
hidden_layer_size = 512
batch_size = 32

learning_rate = LR
keep_probability = KP
regularizer_type = RT
regularization = RN
max_norm_constraint = MN

logging.info('Hyperparameters are:')
logging.info('Learning Rate: %f', learning_rate)
logging.info('Keep Probability: %f', keep_probability)
logging.info('Regularizer Type: %s', regularizer_type)
logging.info('Regularization: %f', regularization)
logging.info('Max Norm Constraint: %d', max_norm_constraint)


# ### Build the network

# In[6]:


# Reset
tf.reset_default_graph()

# Placeholders
x = tf.placeholder(tf.float32, [None, len(X_train.columns)], 'x')
y = tf.placeholder(tf.float32, [None, len(y_train.columns)], 'y')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')

# Regularization
regularizer = None
if regularizer_type == 'L1':
    regularizer = tf.contrib.layers.l1_regularizer(regularization)
elif regularizer_type == 'L2':
    regularizer = tf.contrib.layers.l2_regularizer(regularization)
#TODO - max norm constraint?

# Model
model = tf.contrib.layers.fully_connected(x, hidden_layer_size, tf.nn.relu, 
                                          weights_regularizer=regularizer)
model = tf.nn.dropout(model, keep_prob)
model = tf.contrib.layers.fully_connected(model, hidden_layer_size, tf.nn.relu, 
                                          weights_regularizer=regularizer)
model = tf.nn.dropout(model, keep_prob)
model = tf.contrib.layers.fully_connected(model, len(y_train.columns), 
                                          weights_regularizer=regularizer)
logits = tf.nn.softmax(model)

# Loss and Optimizer
loss = tf.reduce_mean(tf.squared_difference(logits, y))
tf.summary.scalar('loss', loss)
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
tf.summary.scalar('acc', accuracy)

summary_op = tf.summary.merge_all()


# ### For saving the model and writing to Tensorboard

# In[7]:


save_model_path = 'model/'
model_name = 'model'
tensorboard_path = 'tensorboard/'

clear_model(save_model_path, tensorboard_path)

train_summary_writer = tf.summary.FileWriter(tensorboard_path + 'train')
test_summary_writer = tf.summary.FileWriter(tensorboard_path + 'test')

saver = tf.train.Saver()


# ### Training the model

# In[8]:


print 'Training...'

start = datetime.datetime.now()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        n_batches = len(X_train.index) / batch_size
        for batch_i in range(n_batches):
            batch_features, batch_labels = get_batch(X_train, y_train, batch_i, batch_size)
            sess.run(optimizer, feed_dict={ x: batch_features, y: batch_labels, 
                                            keep_prob: keep_probability })
        loss_res, accuracy_res = print_stats(sess, epoch, X_train, y_train, X_validate, 
                                             y_validate, loss, accuracy, x, y, keep_prob, 
                                             summary_op, train_summary_writer, 
                                             test_summary_writer)
        saver.save(sess, save_model_path + model_name, global_step=epoch)
    saver.save(sess, save_model_path + model_name)  
total = datetime.datetime.now() - start

print 'Training complete\nTotal time: %.2f' % total.total_seconds() + ' seconds'
logging.info('Loss: %.2f, Accuracy: %.2f', loss_res, accuracy_res)


# ### Test the model

# In[ ]:


print 'Testing...'

loaded_graph = tf.Graph()

with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(save_model_path + model_name + '.meta')
    loader.restore(sess, save_model_path + model_name)
    loaded_x = loaded_graph.get_tensor_by_name('x:0')
    loaded_y = loaded_graph.get_tensor_by_name('y:0')
    loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
    test_batch_acc_total = 0
    test_batch_count = 0
    n_batches = len(X_test.index) / batch_size
    for batch_i in range(n_batches):
        batch_features, batch_labels = get_batch(X_test, y_test, batch_i, batch_size)
        test_batch_acc_total += sess.run(loaded_acc, feed_dict={ loaded_x: batch_features, 
                                                     loaded_y: batch_labels, 
                                                     loaded_keep_prob: 1.0 })
        test_batch_count += 1
    logging.info('Testing Accuracy: %.2f', test_batch_acc_total / test_batch_count)

