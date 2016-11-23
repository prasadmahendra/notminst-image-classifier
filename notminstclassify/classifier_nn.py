import os
import pickle
import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from notminstclassify.training.trainingdata import TrainingImageData

def load_training_data_if():
    saved_training_and_test_data = 'notMNIST-training-data.pickle'
    if not os.path.isfile(saved_training_and_test_data):
        # Get the features and labels from the zip files
        trainingdata = TrainingImageData('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', '../data/training/notMNIST_train.zip')
        trainingdata.load()
        trainingdata.resample(size_limit=150000) # Limit the amount of data to work with a docker container
        trainingdata.normalize_features()
        trainingdata.hot_one_encode_labels()
        trainingdata.assert_is_features_normal()
        trainingdata.assert_is_labels_encod()

        testdata = TrainingImageData('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', '../data/training/notMNIST_test.zip')
        testdata.load()
        testdata.normalize_features()
        testdata.hot_one_encode_labels()
        testdata.assert_is_features_normal()
        testdata.assert_is_labels_encod()

        # Get randomized datasets for training and validation
        train_features, valid_features, train_labels, valid_labels = trainingdata.train_test_split()
        test_features = testdata.features
        test_labels = testdata.labels

        # Save the data for easy access
        print('Saving data to pickle file...')
        try:
            with open(saved_training_and_test_data, 'wb') as pfile:
                pickle.dump(
                    {
                        'train_dataset': train_features,
                        'train_labels': train_labels,
                        'valid_dataset': valid_features,
                        'valid_labels': valid_labels,
                        'test_dataset': test_features,
                        'test_labels': test_labels
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', saved_training_and_test_data, ':', e)
            raise

        print('Data cached in pickle file.')
    else:
        with open(saved_training_and_test_data, 'rb') as f:
            pickle_data = pickle.load(f)
            train_features = pickle_data['train_dataset']
            train_labels = pickle_data['train_labels']
            valid_features = pickle_data['valid_dataset']
            valid_labels = pickle_data['valid_labels']
            test_features = pickle_data['test_dataset']
            test_labels = pickle_data['test_labels']
            del pickle_data  # Free up memory

        print('Data and modules loaded.')

    return  train_features, valid_features, test_features, train_labels, valid_labels, test_labels


def train():
    train_features, valid_features, test_features, train_labels, valid_labels, test_labels = load_training_data_if()

    # notMINST dataset contains images of size 28x28. Our feature vector is there a 1x784 vector
    features_count = 784
    # notMINST dataset contains 10 classes of images ...
    labels_count = 10

    # build our training NN
    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)

    weights = tf.Variable(tf.truncated_normal(shape=[features_count,labels_count], dtype=tf.float32))
    biases = tf.Variable(tf.zeros(shape=[labels_count], dtype=tf.float32))

    # Feed dicts for training, validation, and test session
    train_feed_dict = {features: train_features, labels: train_labels}
    valid_feed_dict = {features: valid_features, labels: valid_labels}
    test_feed_dict = {features: test_features, labels: test_labels}

    # NN
    # X -> Linear Func -> SoftMax -> Cross entropy

    # Linear Function WX + b
    logits = tf.matmul(features, weights) + biases
    # softmax
    prediction = tf.nn.softmax(logits)
    # Cross entropy
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)

    # Training loss
    loss = tf.reduce_mean(cross_entropy)

    # Create an operation that initializes all variables
    init = tf.initialize_all_variables()

    # Test Cases
    with tf.Session() as session:
        session.run(init)
        session.run(loss, feed_dict=train_feed_dict)
        session.run(loss, feed_dict=valid_feed_dict)
        session.run(loss, feed_dict=test_feed_dict)
        biases_data = session.run(biases)

    assert not np.count_nonzero(biases_data), 'biases must be zeros'

    print('NN Created!')

    # Determine if the predictions are correct
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    # Calculate the accuracy of the predictions
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))

    print('Accuracy function created.')

    # Tweak: Find the best parameters for each configuration
    epochs = 5
    batch_size = 100
    learning_rate = 0.2

    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # The accuracy measured against the validation set
    validation_accuracy = 0.0

    # Measurements use for graphing loss and accuracy
    log_batch_step = 50
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []

    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(train_features) / batch_size))

        for epoch_i in range(epochs):

            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i * batch_size
                batch_features = train_features[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer and get loss
                _, l = session.run(
                    [optimizer, loss],
                    feed_dict={features: batch_features, labels: batch_labels})

                # Log every 50 batches
                if not batch_i % log_batch_step:
                    # Calculate Training and Validation accuracy
                    training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                    validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

                    # Log batches
                    previous_batch = batches[-1] if batches else 0
                    batches.append(log_batch_step + previous_batch)
                    loss_batch.append(l)
                    train_acc_batch.append(training_accuracy)
                    valid_acc_batch.append(validation_accuracy)

            # Check accuracy against Validation data
            validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)

    loss_plot = plt.subplot(211)
    loss_plot.set_title('Loss')
    loss_plot.plot(batches, loss_batch, 'g')
    loss_plot.set_xlim([batches[0], batches[-1]])
    acc_plot = plt.subplot(212)
    acc_plot.set_title('Accuracy')
    acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
    acc_plot.plot(batches, valid_acc_batch, 'b', label='Validation Accuracy')
    acc_plot.set_ylim([0, 1.0])
    acc_plot.set_xlim([batches[0], batches[-1]])
    acc_plot.legend(loc=4)
    plt.tight_layout()
    #plt.show()

    print('Validation accuracy at {}'.format(validation_accuracy))

    # Tweak: Set the epochs, batch_size, and learning_rate with the best parameters from above (training gradient descent)
    epochs = 5
    batch_size = 100
    learning_rate = 0.2

    # The accuracy measured against the test set
    test_accuracy = 0.0

    with tf.Session() as session:
        session.run(init)
        batch_count = int(math.ceil(len(train_features) / batch_size))

        for epoch_i in range(epochs):

            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i + 1, epochs), unit='batches')

            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i * batch_size
                batch_features = train_features[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]

                # Run optimizer
                _ = session.run(optimizer, feed_dict={features: batch_features, labels: batch_labels})

            # Check accuracy against Test data
            test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)

    assert test_accuracy >= 0.80, 'Test accuracy at {}, should be equal to or greater than 0.80'.format(test_accuracy)
    print('Test Accuracy is {}'.format(test_accuracy))

train()