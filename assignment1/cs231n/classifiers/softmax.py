import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    
  num_train = X.shape[0]
  pix = X.shape[1]
  num_class = W.shape[1]

  for i in range(num_train):
        Xi = np.array([X[i,:]])        
        score_i = Xi.dot(W)
        score_i -= np.max(score_i) #for numeric stability
        exp_row = np.exp(score_i)
        sum_exp = np.sum(exp_row)
        loss += -1*np.log(exp_row[0,y[i]]/sum_exp)
        
        exp_row[0,y[i]]-=sum_exp
        Li = Xi.T.dot(exp_row)
        Li/=sum_exp
        dW +=Li
  
  dW/=num_train
  dW+=reg*2*W
        
        
    
  loss/=num_train
  loss+=reg*np.sum(W*W)
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
    
  num_train = X.shape[0]
  num_pix = X.shape[1]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  scores -= scores.max(axis=1)[:,np.newaxis] #for numerical stability
  exp_scores = np.exp(scores)
  
  row_sum = np.sum(exp_scores,axis=1)
  fract = exp_scores[range(num_train),y]/row_sum
  loss = np.sum(-1*np.log(fract),axis=0)
  loss/=num_train
  loss+=reg*np.sum(W*W)
    
  #computing the gradient
  exp_scores[range(num_train),y] -= row_sum
  dW = (X.T/row_sum).dot(exp_scores)
  dW/=num_train
  dW+=reg*2*W
  
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

