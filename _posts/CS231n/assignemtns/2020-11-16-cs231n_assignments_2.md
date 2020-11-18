---
layout: post
title: "cs231n assignment 2"
comments: true
categories: cs231n
---

이 포스트에선 cs231n 강좌의 assignment 2에 대해서 다룬다.

## Q1: FullyConnectedNets

이 과제의 핵심은 modular이다. 기존에 우리가 코드를 작성했던 방식은 잘 돌아가긴 했지만, 다른 architecture에 적용하기가 힘들었다. input-output만 보았을 때에 단순히 모든 계산이 끝나져서 나오니까... 따라서 보다 더 modular하게 코드를 작성하려면, path를 각각 분리해서 작성할 필요가 있다.

```python
def layer_forward(x, w):
  """ Receive inputs x and weights w """
  # Do some computations ...
  z = # ... some intermediate value
  # Do some more computations ...
  out = # the output
   
  cache = (x, w, z, out) # Values we need to compute gradients
   
  return out, cache
```
이처럼 loss를 직접적으로 계산하는 forward path와,

```python
def layer_backward(dout, cache):
  """
  Receive dout (derivative of loss with respect to outputs) and cache,
  and compute derivative with respect to inputs.
  """
  # Unpack cache values
  x, w, z, out = cache
  
  # Use values in cache to compute derivatives
  dx = # Derivative of loss with respect to x
  dw = # Derivative of loss with respect to w
  
  return dx, dw
```

해당 loss에 대한 gradient를 계산하는 backward path.

만약 우리가 특정 layer에 대해서 두 path를 가지고 있다면, 단순히 layer를 stack하는것만으로도 손쉽게 gradient를 계산할 수 있을 것이다.

이를 바탕으로 먼저 fc layer를 implement해보자.