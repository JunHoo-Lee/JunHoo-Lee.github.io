---
layout: post
title: "Python tutorial"
comments: true
categories: python
---

## Numpy broadcasting

[broadcast](https://numpy.org/doc/stable/user/basics.broadcasting.html)

## Advanced indexing

```python
>>> x = np.array([[1, 2], [3, 4], [5, 6]])
>>> x[[0, 1, 2], [0, 1, 0]]
array([1, 4, 5])
```

[index numpy](https://numpy.org/doc/stable/reference/arrays.indexing.html)

## numpy.sum

```python
>>> np.sum([0.5, 1.5])
2.0
>>> np.sum([0.5, 0.7, 0.2, 1.5], dtype=np.int32)
1
>>> np.sum([[0, 1], [0, 5]])
6
>>> np.sum([[0, 1], [0, 5]], axis=0)
array([0, 6])
>>> np.sum([[0, 1], [0, 5]], axis=1)
array([1, 5])
>>> np.sum([[0, 1], [np.nan, 5]], where=[False, True], axis=1)
array([1., 5.])
```

[numpy sum](https://numpy.org/doc/stable/reference/generated/numpy.sum.html)

## Change dimension order

```python
>>> x = np.zeros((3, 4, 5))
>>> np.moveaxis(x, 0, -1).shape
(4, 5, 3)
>>> np.moveaxis(x, -1, 0).shape
(5, 3, 4)
>>> np.transpose(x).shape
(5, 4, 3)
>>> np.swapaxes(x, 0, -1).shape
(5, 4, 3)
>>> np.moveaxis(x, [0, 1], [-1, -2]).shape
(5, 4, 3)
>>> np.moveaxis(x, [0, 1, 2], [-1, -2, -3]).shape
(5, 4, 3)

```

## Elememtwise multiplication

```python
>>> x1 = np.arange(9.0).reshape((3, 3))
>>> x2 = np.arange(3.0)
>>> np.multiply(x1, x2)
array([[  0.,   1.,   4.],
       [  0.,   4.,  10.],
       [  0.,   7.,  16.]])
```

[np.multiply](
https://numpy.org/doc/stable/reference/generated/numpy.multiply.html)

## Add shape

```python
a = np.array([[1,2,3],[1,1,1]])

b = np.ones(shape = (a.shape + (2,)))
b
>>> array([[[1., 1.],
        [1., 1.],
        [1., 1.]],

       [[1., 1.],
        [1., 1.],
        [1., 1.]]])
```

## numpy.random.choice

used for making batches

[numpy.random.choice](https://numpy.org/doc/stable/reference/random/generated/numpy.random.choice.html)

```python
>>> np.random.choice(5, 3)
array([0, 3, 4]) # random
>>>  #This is equivalent to np.random.randint(0,5,3)

>>>  #Generate a non-uniform random sample from np.arange(5) of size 3:

>>> np.random.choice(5, 3, p=[0.1, 0, 0.3, 0.6, 0])
array([3, 3, 0]) # random

>>> #Generate a uniform random sample from np.arange(5) of size 3 without replacement:

>>> np.random.choice(5, 3, replace=False)
array([3,1,0]) # random
>>>  #This is equivalent to np.random.permutation(np.arange(5))[:3]
>>>  #Generate a non-uniform random sample from np.arange(5) of size 3 without replacement:

>>> np.random.choice(5, 3, replace=False, p=[0.1, 0, 0.3, 0.6, 0])
array([2, 3, 0]) # random
>>> #Any of the above can be repeated with an arbitrary array-like instead of just integers. For instance:

>>> aa_milne_arr = ['pooh', 'rabbit', 'piglet', 'Christopher']
>>>  np.random.choice(aa_milne_arr, 5, p=[0.5, 0.1, 0.1, 0.3])
array(['pooh', 'pooh', 'pooh', 'Christopher', 'piglet'], # random
      dtype='<U11')
```

## tqdm

it shows for loop progressing in graphic version

[tqdm](https://github.com/tqdm/tqdm)

```python
from tqdm import tqdm
for i in tqdm(range(10000)):
    ...
```

## numpy.amax

if you want to get element which has maximun value in desired array...

```python
>>> a = np.arange(4).reshape((2,2))
>>> a
array([[0, 1],
       [2, 3]])
>>> np.amax(a)           # Maximum of the flattened array
3
>>> np.amax(a, axis=0)   # Maxima along the first axis
array([2, 3])
>>> np.amax(a, axis=1)   # Maxima along the second axis
array([1, 3])
>>> np.amax(a, where=[False, True], initial=-1, axis=0)
array([-1,  3])
>>> b = np.arange(5, dtype=float)
>>> b[2] = np.NaN
>>> np.amax(b)
nan
>>> np.amax(b, where=~np.isnan(b), initial=-1)
4.0
>>> np.nanmax(b)
4.0
```

## numpy.argmax

if you want to get index which has maximum value in desired array...

```python
>>> a = np.arange(6).reshape(2,3) + 10
>>> a
array([[10, 11, 12],
       [13, 14, 15]])
>>> np.argmax(a)
5
>>> np.argmax(a, axis=0)
array([1, 1, 1])
>>> np.argmax(a, axis=1)
array([2, 2])
```
## numpy.where
numpy.where(condition[, x, y])
Return elements chosen from x or y depending on condition.

[numpy.where](https://numpy.org/doc/stable/reference/generated/numpy.where.html)

### Parameters

conditionarray_like, bool
Where True, yield x, otherwise yield y.

x, y : array_like

Values from which to choose. x, y and condition need to be broadcastable to some shape.

### Returns

out : ndarray

An array with elements from x where condition is True, and elements from y elsewhere.

```python
>>> a = np.array([[0, 1, 2],
                  [0, 2, 4],
                  [0, 3, 6]])
>>> np.where(a < 4, a, -1)  # -1 is broadcast

array([[ 0,  1,  2],
       [ 0,  2, -1],
       [ 0,  3, -1]])
```


## np_SPACE

np.linspace 

numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)
[np.linapce](https://numpy.org/doc/stable/reference/generated/numpy.linspace.html#numpy.linspace)

np.geomspace

numpy.geomspace(start, stop, num=50, endpoint=True, dtype=None, axis=0)
[np.geomspace](https://numpy.org/doc/stable/reference/generated/numpy.geomspace.html#numpy.geomspace)
