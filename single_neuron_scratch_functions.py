import matplotlib.pyplot as plt
import numpy as np
import math

def load_flower_dataset(num_samples=500, petals=4, petal_length=4, noise=0.2, angle=30):
    '''
    Create synthetic flower 2D dataset with two classes(0/1)
    
    Parameters
    ----------
    num_samples : int (default=500)
        number of overall samples to be created
    petals : int (default=4)
        represents half the number of petals of flower
    petal_length : int (default=4)
    noise : float (default=0.2)
    angle : int (default=30)
        angle in degrees to couter-clockwise rotate the flower
    
    Returns
    -------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1
    '''
    np.random.seed(1)
    m = num_samples # number of examples
    N = int(m/2) # number of points per class
    D = 2 # dimensionality
    X = np.zeros((m,D)) # data matrix where each row is a single example
    Y = np.zeros((m,1), dtype='uint8') # labels vector (0 for red, 1 for blue)

    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*np.pi,(j+1)*np.pi,N) + np.random.randn(N)*noise # theta, classes mixing imperfection
        r = petal_length*np.sin(petals*t) + np.random.randn(N)*noise # radius, petal shape imperfection
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        Y[ix] = j

    # rotating data points
    theta = np.radians(angle)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
    X = np.dot(X, rotation_matrix)
    # previous shape is (num_samples, dim), we need to transpose it to (dim, num_samples
    X = X.T
    # previous shape is (num_samples,), we need to reshape it to (1, num_samples)
    Y = Y.T

    return X, Y


def epoch(X, Y, w, b, learning_rate=0.05):
    '''
    Perform one epoch(cycle) of training for NEURON[logistic regression]
    
    Parameters
    ----------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1
    w : numpy.ndarray [shape: (#features, 1)]
        array containing weights used by neuron
    b : float
        bias used by neuron
    learning_rate : float (default=0.05)   

    Returns
    -------
    w : updated array of learned weights by neuron
    b : updated learned bias by neuron
    dw : array containing increments that were added to weights
    db : increment that was added to bias
    cost: average loss of samples with input parameters
    '''    
    m = X.shape[1]
    dw = np.zeros((X.shape[0],1))
    db = 0
    cost=0
    for i in range(m):
        # FORWARD PROPAGATION
        z = np.dot(w.T,X[:,i])+b
        A = 1/(1+math.exp(-z))

        # BACKWARD PROPAGATION (ADDING COST FUNCTION)
        cost = cost + (-( Y[0,i]*np.log(A)+(1-Y[0,i])*np.log(1-A)))
        # BACKWARD PROPAGATION (ADDING GRADS)
        dz = A-Y[0,i]
        dw = dw + np.dot(X[:,i].reshape(-1,1),dz)
        db = db + dz

    # BACKWARD PROPAGATION (FINDING MEAN)
    cost = 1/m*cost
    dw = 1/m*dw
    db = 1/m*db

    # UPDATE PARAMETERS
    w = w - learning_rate*dw
    b = b- learning_rate*db
    
    return w, b, dw, db, cost


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using NEURON[learned logistic regression] parameters (w, b)
    
    Parameters
    ----------
    w : numpy.ndarray [shape: (#features, 1)]
        array containing weights used by neuron
    b : float
        bias used by neuron
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data

    Returns
    -------
    Y_prediction: numpy.ndarray [shape: (1, #samples)]
        array containing predictions 0 or 1
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(-1, 1)
    
    # Compute forward propagation
    z = np.matmul(w.T,X)+b
    A = 1/(1+np.exp(-z))
    
    # Convert probabilities A[0,i] to actual predictions p[0,i]
    for i in range(A.shape[1]):
        if A[0, i] > 0.5 :
            Y_prediction[0,i] = 1
        else:
            Y_prediction[0,i] = 0
    
    return Y_prediction


def plot_scatter(X,Y):
    '''
    Show the scatter plot of flower dataset
    
    Parameters
    ----------
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1 
    '''
    scatter=plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
    plt.legend(*scatter.legend_elements(), loc="upper right", title="Classes")
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.show()


def plot_decision_boundary(w, b, X, Y):
    """
    Plot the decision boundary for logistic regression
    
    Parameters
    ----------
    w : numpy.ndarray [shape: (#features, 1)]
        array containing weights used by neuron
    b : float
        bias used by neuron
    X : numpy.ndarray [shape: (#features, #samples)]
        matrix of data
    Y : numpy.ndarray [shape: (1, #samples)]
        array containing true labels 0 or 1 
    """
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = predict(w,b,np.c_[xx.ravel(), yy.ravel()].T)
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plot_scatter(X,Y)


