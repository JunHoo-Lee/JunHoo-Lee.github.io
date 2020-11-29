---
layout: post
title: "Towards Evaluating the Robustness of Neural Networks"
categories: mipal
---
## Neural Networks Notation

$$
F = softmax\cdot F_n \cdot F_{n-1} \cdots F_{1}
$$


## Adverserial Examples

그니까 우리가 어떤 input x에 대해서 $C^*(x)$(x의 real label data)를 안다고 하자. 이때, $t \neq C^*(x)$인 t에 대해서 $t = C(\tilde{x})$를 만족하는 $x$와 metric이 매우 가까운 $\tilde{x}$를 찾는것이 이 문제이다.

어떨 때는 target 없이 단순히 $C(\tilde{x}) \neq C^*(x)$를 찾는 Untargeted attack도 수행하긴 하지만, 본 논문에선 targeted attack만 찾는다.

이 세가지 방식으로 Target을 생각하는데,

1. Average Case: uniformly하게 target을 택하여 결과값을 보는 것
2. Best Case: 가장 train 하기 쉬운 class에 대해서 결과를 report하는 것
3. Worst Case: 그 반대

## Distance Metric

다음과 같이 생각한다
$$
||v||_p = \left(\sum_{i=1}^n|v_i|^p\right)^{\frac{1}{p}}
$$

특이한 경우만 생각해보자면, 

$L_0$ distance를 생각해 보자. 만약 $x_i \neq \tilde{x}^i$라면 $||v||^i_p = 1$이 될 것이다, 같다면 0이 될 것이고...즉 이 $L_0$ distance는 pixel이 몇개나 바뀌었는지를 보여준다.

$L^\infty$ distance를 생각한다면, 
$$L^\infty = max(|x_1 - \tilde{x}_1|,|x_2 - \tilde{x}_2| \dotsc, |x_n - \tilde{x}_n| )$$

이고, 즉 이것은 이미지를 바꿀 때 한 이미지 픽셀을 바꿀 수 있는 maximum 변화량을 의미한다.

## Defensive Distillation

![KD](/images/2020-11-23-13-01-32.png)

즉 label data를 학습시킬 때, 제공된 label data를 쓰지 않고 기존에 train했던 model의 smoothe label data를 사용하는 것이다. 결국 이 과정을 통해서 training data의 overfitting을 막는것이 그 목적인데, highly nonlinear한 곳의 blind spot 때문에 이런 문제가 발생한다고 가정하고, 위의 방식을 수행함으로서 막는 것이다.

## Attack Algorithms

### L- BFGS

다음과 같은 $x^\prime$을 찾는것이 목적이다

$$
\text{minimize } ||x-x^\prime||^2_2\\
\text{such that }C(x^\prime)=l\\
x^\prime\in[0,1]^n
$$

하지만 당연히 이런 문제를 푸는것은 굉장히 어려운 일이고, 다음과 같이 문제를 변형해서 푼다.

$$
\text{minimize } c \cdot||x-x^\prime||^2_2 + loss_{F,l}(x^\prime)\\
\text{such that }
x^\prime\in[0,1]^n
$$
여기서 loss function은 우리가 일반적으로 생각하는 deep learning에서 cross entropy나 softmax같은거

## Approach in This papaer

이런 big picture에서 시작한다

$$
\text{minimize }D(x,x+\delta)\\
\text{such that }C(x+\delta) = t\\
x+\delta \in [0,1]^n
$$
여기서 D는 distance metric이다.

이때 D를 $l_p$ norm이라고 하면,

이 문제는

![ㅇㅇ](/images/2020-11-23-13-42-27.png)

$$
\text{minimize }||\delta||_p + cf(x+\delta)\\
\text{such that }
x+\delta \in [0,1]^n
$$

로 변형될 수 있다

이제 c는, 경험적으로 우리의 솔루션 $x^*$이 $f(x^*) \leq 0$이 되는 minimum c를 잡는게 좋다고 한다

### Box Constraints

당연히 색은 [0,255], clip한다면 [0,1]의 값을 가질 수밖에 없다. 하지만 우리가 gradient를 바꾸면서 [0,1] 이상의 값을 가질 수도 있다. 그래서 이것을 해결하기 위해서 여러가지 방법들을 사용한다.

## Question

1. blind spot?
2. 굉장히 간단한 모델을 사용하는데 그 이유가 ResNet나 googlenet같은 아키텍쳐는 당연히 더 약하기 때문인지?
3. highly non linear하다는 게 정확하게 무슨 말인지. deep learning은 linear 아닌가? activation layer때문에?
4. Objective function에서 0보다 작을 수 없는 function이 있는것 같은데 f2같은건 이미 필요충분조건 아닌가? F(x`) = t가 나오면 저 조건 자연스럽게 만족하는거 아닌가?
5. 정확하게 새로운 방식이 기존 방식과 어떻게 다른지 모르겠다

![objective ](/images/2020-11-23-13-32-08.png)

x+\delta \in [0,1]^n 이거 내가 생각하는 의미 맞나