# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import util
from pacman import GameState
import random
import numpy as np
from pacman import Directions
import math
import numpy as np
from featureExtractors import FEATURE_NAMES

PRINT = True


class PerceptronPacman:

    def __init__(self, num_train_iterations=20, learning_rate=1):

        self.max_iterations = num_train_iterations
        self.learning_rate = learning_rate

        # A list of which features to include by name. To exclude a feature comment out the line with that feature name
        feature_names_to_use = [
            'closestFood', 
            'closestFoodNow',
            'closestGhost',
            'closestGhostNow',
            'closestScaredGhost',
            'closestScaredGhostNow',
            'eatenByGhost',
            'eatsCapsule',
            'eatsFood',
            "foodCount",
            'foodWithinFiveSpaces',
            'foodWithinNineSpaces',
            'foodWithinThreeSpaces',  
            'furthestFood', 
            'numberAvailableActions',
            "ratioCapsuleDistance",
            "ratioFoodDistance",
            "ratioGhostDistance",
            "ratioScaredGhostDistance"
            ]
        
        # we start our indexing from 1 because the bias term is at index 0 in the data set
        feature_name_to_idx = dict(zip(FEATURE_NAMES, np.arange(1, len(FEATURE_NAMES) + 1)))

        # a list of the indices for the features that should be used. We always include 0 for the bias term.
        self.features_to_use = [0] + [feature_name_to_idx[feature_name] for feature_name in feature_names_to_use]

        "*** YOUR CODE HERE ***"
        self.N_input = len(self.features_to_use)  # Number of input features (including bias term)
        self.N_hidden = 50  # Number of neurons in the hidden layer
        self.N_output = 1   # Number of output neurons

        # Initialize weights
        # Weights between input layer and hidden layer
        self.W1 = np.random.randn(self.N_input, self.N_hidden) * 0.01  # Shape: (N_input x N_hidden)
        # Weights between hidden layer and output layer
        self.W2 = np.random.randn(self.N_hidden, self.N_output) * 0.01  # Shape: (N_hidden x N_output)

    def predict(self, feature_vector):
        """
        This function should take a feature vector as a numpy array and pass it through your perceptron and output activation function

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.
        """
        # filter the data to only include your chosen features. We might not need to do this if we're working with training data that has already been filtered.
        if len(feature_vector) > len(self.features_to_use):
            vector_to_classify = feature_vector[self.features_to_use]
        else:
            vector_to_classify = feature_vector

        "*** YOUR CODE HERE ***"
        # Forward pass
        # Input to hidden layer
        z1 = np.dot(vector_to_classify, self.W1)  # Shape: (N_hidden, )
        # Apply activation function for hidden layer
        a1 = self.activationHidden(z1)  # Shape: (N_hidden, )
        # Hidden to output layer
        z2 = np.dot(a1, self.W2)  # Shape: (N_output, )
        # Apply activation function for output layer
        y_hat = self.activationOutput(z2)  # Shape: (N_output, )

        return y_hat[0]  # Return scalar value since N_output = 1


    def activationHidden(self, x):
        """
        Implement your chosen activation function for any hidden layers here.
        """

        "*** YOUR CODE HERE ***"
        return np.maximum(0, x)  # ReLU activation
    
    

    def activationOutput(self, x):
        """
        Implement your chosen activation function for the output here.
        """

        "*** YOUR CODE HERE ***"
        return 1 / (1 + np.exp(-x))  # Sigmoid activation

    def evaluate(self, data, labels):
        """
        This function should take a data set and corresponding labels and compute the performance of the perceptron.
        You might for example use accuracy for classification, but you can implement whatever performance measure
        you think is suitable. You aren't evaluated what you choose here. 
        This function is just used for you to assess the performance of your training.

        The data should be a 2D numpy array where each row is a feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, labels[1]
        is the label for the feature at data[1]
        """

        # filter the data to only include your chosen features
        X_eval = data[:, self.features_to_use]

        "*** YOUR CODE HERE ***"
        X = data[:, self.features_to_use]
        y_true = labels.values  # Assuming labels are pandas Series

        num_samples = X.shape[0]
        total_loss = 0.0
        correct_predictions = 0

        for i in range(num_samples):
            x = X[i]
            y_t = y_true[i]

            # Forward pass
            z1 = np.dot(x, self.W1)
            a1 = self.activationHidden(z1)
            z2 = np.dot(a1, self.W2)
            y_hat = self.activationOutput(z2)

            # Compute loss
            epsilon = 1e-7  # For numerical stability
            y_hat_clipped = np.clip(y_hat, epsilon, 1 - epsilon)
            loss = - (y_t * np.log(y_hat_clipped) + (1 - y_t) * np.log(1 - y_hat_clipped))
            total_loss += loss

            # Compute accuracy
            prediction = 1 if y_hat >= 0.5 else 0
            if prediction == y_t:
                correct_predictions += 1

        avg_loss = total_loss / num_samples
        accuracy = correct_predictions / num_samples

        print(f'Loss: {avg_loss}, Accuracy: {accuracy}')
        return avg_loss, accuracy


    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        This function should take training and validation data sets and train the perceptron

        The training and validation data sets should be 2D numpy arrays where each row is a different feature vector

        THE FEATURE VECTOR WILL HAVE AN ENTRY FOR BIAS ALREADY AT INDEX 0.

        The training and validation labels should be a list of 1s and 0s, where the value at index i is the
        corresponding label for the feature vector at index i in the appropriate data set. For example, trainingLabels[1]
        is the label for the feature at trainingData[1]
        """

        # filter the data to only include your chosen features. Use the validation data however you like.
        X_train = trainingData[:, self.features_to_use]
        

        "*** YOUR CODE HERE ***"
        y_train = trainingLabels.values  # Assuming labels are pandas Series

        num_samples = X_train.shape[0]

        for epoch in range(self.max_iterations):
            # Shuffle the training data each epoch
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(num_samples):
                x = X_train[i]  # Input vector
                y_true = y_train[i]  # True label (0 or 1)

                # Forward pass
                z1 = np.dot(x, self.W1)
                a1 = self.activationHidden(z1)
                z2 = np.dot(a1, self.W2)
                y_hat = self.activationOutput(z2)

                # Backpropagation
                # Compute output layer error
                delta2 = y_hat - y_true  # Shape: scalar

                # Compute gradient for W2
                dW2 = np.outer(a1, delta2)  # Shape: (N_hidden, N_output)

                # Compute hidden layer error
                dz1 = (z1 > 0).astype(float)  # Derivative of ReLU
                delta1 = dz1 * (self.W2[:, 0] * delta2)  # Shape: (N_hidden, )

                # Compute gradient for W1
                dW1 = np.outer(x, delta1)  # Shape: (N_input, N_hidden)

                # Update weights
                self.W2 -= self.learning_rate * dW2
                self.W1 -= self.learning_rate * dW1

            # Evaluate on validation data
            val_loss, val_accuracy = self.evaluate(validationData, validationLabels)
            print(f'Epoch {epoch + 1}/{self.max_iterations}, Validation Loss: {val_loss}, Validation Accuracy: {val_accuracy}')
        

    

    def save_weights(self, weights_path):
        """
        Saves your weights to a .model file. You're free to format this however you like.
        For example with a single layer perceptron you could just save a single line with all the weights.
        """
        "*** YOUR CODE HERE ***"
        W1 = self.W1
        W2 = self.W2
        with open(weights_path, "wb") as f:
            np.save(f, W1)
            np.save(f, W2)

    def load_weights(self, weights_path):
        """
        Loads your weights from a .model file. 
        Whatever you do here should work with the formatting of your save_weights function.
        """
        "*** YOUR CODE HERE ***"
        with open(weights_path, "rb") as f:
            self.W1 = np.load(f)
            self.W2 = np.load(f)