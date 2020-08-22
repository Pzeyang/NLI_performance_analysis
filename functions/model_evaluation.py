###functions for evaluating the deep learning models
import sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, accuracy_score
from scipy.stats import spearmanr, pearsonr
import numpy as np
import pandas as pd

def evaluate_model(y_true, y_pred):
    """Function to calculate the precision, recall, f1_score and accuracy of the model
        Args:
            y_true: actual value of the classes
            y_pred: predicted value of the classes"""
    prec=  precision_score(y_true, y_pred, average = 'weighted')
    rec = recall_score(y_true, y_pred, average = 'weighted')
    f1_s = f1_score(y_true, y_pred, average = 'weighted')
    acc= accuracy_score(y_true, y_pred)
    return prec,rec,f1_s, acc

def plot_model_training(hist):
    """Function to plot accuracy and loss for iterations for the training and validation data set of the model
       Args:
           model: trained model"""
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

    # Plot loss
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')

def evaluate_relatedness(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    pearson = pearsonr(y_pred, y_true)
    spearman = spearmanr(y_pred, y_true)
    mean_absolute_deviation = np.mean(np.abs((y_true - y_pred)/y_true))*100    
    return pearson, spearman, mean_absolute_deviation
