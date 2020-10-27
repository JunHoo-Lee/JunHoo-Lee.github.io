# CS231n assignment 1

## Linear SVM

```python
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,j] += X[i]
                dW[:,y[i]] -= X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2*reg*W

```

The first question is to implement gradient with given loss.

When we think its backpropagation, it should be like

![svm_grad](./image/svm_gradient.png)

So without regression, its gradient should be

1. For correct class, (#classes_should_be_edited)*(X)
2. For should_be_corrected_classes(which score exceeds Cor_score + 1), X

and with gradient, 2W * Reg

## Fully vectorized algorithm

```python
tot_score = X.dot(W)
    correct_class_score = np.zeros(X.shape[0])
    correct_class_score = tot_score[np.arange(X.shape[0]),y]
    # make all elements score - correct_class_score + 1
    tot_score -= correct_class_score[:,np.newaxis]
    tot_score += 1
    # We don't regard right class margin
    tot_score[np.arange(X.shape[0]),y] = 0
    # Find below_zero_elements(apply max function)
    tot_score[tot_score <= 0] = 0
    # calculate loss
    loss = np.sum(tot_score)/(X.shape[0])
    # add regularlization term
    loss += reg * np.sum(W * W)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    #Set gradient which Dimension is N x C x D
    grad_large = np.zeros(shape = (tot_score.shape +(X.shape[1],)))    
    #indicate which col should be calculated (i.e. score + 1- cor_score > 0)
    grad_large[tot_score > 0] = 1
    # calculate how many rows are to be calcuated and its # is the number of modifying correct class row
    #indicate correct score
    grad_large[np.arange(X.shape[0]),y] -= np.sum(grad_large, axis = 1)
    # change its order of Dimension N x C x D to N x D x C
    grad_large = np.swapaxes(grad_large, 2 , 1)
    # make element_wise multiplication to X
    grad_large = np.multiply(grad_large,X[...,np.newaxis])
    # make sum
    dW = np.sum(grad_large,axis = 0)
    # divide by its training number
    dW /= X.shape[0]
    # add regression term
    dW += 2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
```

## make batch of linear_classifier.py
```python
 for it in tqdm(range(num_iters)):
            X_batch = None
            y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            batch_index = np.random.choice(X.shape[0], batch_size)
            X_batch = X[batch_index]
            y_batch = y[batch_index]

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            self.W -= learning_rate * grad

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history
```

it was simple, pseudo code was..

```python
    batch_indices = get_array_indices
    batch = X[batch_indices]
    batch_label = y[batch_indices]
    get loss and by loss function

    update gradient
```

## make prediction

it was more simpleeeeee

```python
    score = X.dot(W)
    ans = np.argmax(score,axis = 1)
```

I used advanced indexing, and broadcasting. 

## Question

1. I want to index with array but it can't
    for example. 

    ``` python
    a = np.array([[1,2],[3,4]])
    a[[0,1],[0,1]]
    >>> 1, 4
    ## but
    b = np.array([[0,1],[0,1]])
    a[b]
    >>> array([[[1, 2],
            [3, 4]],

        [[1, 2],
            [3, 4]]])
    ## ???
    ```

2. why fully-vectorized algorithm doesn't make the runtime fast?

   ```python
    Naive loss and gradient: computed in 0.132488s
    Vectorized loss and gradient: computed in 0.124351s
    difference: 0.000000
   ``` 

3. What is sampling with replacement?

   ```python
    # Hint: Use np.random.choice to generate indices. Sampling with         #
    # replacement is faster than sampling without replacement.              #
   ```

   - solved, it was option of np.ramdom.choice(in Korean, 복원추출 옵션)

4. **Inline Question 1**

    It is possible that once in a while a dimension in the gradcheck will not match exactly. What could such a discrepancy be caused by? Is it a reason for concern? What is a simple example in one dimension where a gradient check could fail? How would change the margin affect of the frequency of this happening? *Hint: the SVM loss function is not strictly speaking differentiable*

    because of my poor english...

![softmax_incorrect](/images/softmax_incorrect.png)
![softmax_correct](/images/softmax_correct.png)