---
layout: post
title: "Logistic Regression"
categories: mipal
---

앞서 우리는 Linear Regression(선형 회기)을 배웠는데, 이는 주어진 종속 변수 Y를 설명 변수 X로 설명하는 방식이었다. 

이는 Y가 continuous하고 unbounded된 변수라면 상관이 없었다. 하지만 Y를 단순히 unbounded되고 continuous한 변수가 아니라, 확률같이 [0,1]로 bounded된 변수라면 이를 어떻게 설명해야 할까?

$$
Y = \theta X
$$

만약, X가 unbounded라면, 위 식으로는 확률을 표현하기에 부적절하다. 1을 넘는 확률이 있을 순 없으니까... 그래서 확률을 추정할 때는 단순한 선형회귀가 아닌, 다른 함수로서의 추정이 필요하다.

![예시](/images/2020-12-11-14-59-21.png)

다음과 같은 bounded s_curve function으로 맞추면 될 것 같다. 이경우엔


$$
\exists f \text{ for } x \in ^{\forall}X s.t.\text{ } 0<f(x)<1
$$

라는 말과 동일하다. 따라서 이제 이런 f를 어떻게 찾을지? 에 대해서 생각해보아야 한다.

## Logit Function

직접적인 f를 찾는것은 직관적으로 어려우니, 그냥 [0,1]를 정의역으로 갖고, $[-\infty,\infty]$를 치역으로 갖는 함수를 찾고, 이에 대한 역함수를 찾아보는 건 어떨까?

p에 대한 로짓 변환은 이를 만족한다.

$$
logit(p) = \log(\frac{p}{1-p})
$$

이 함수에 대해서 역함수를 취해 $logit(p)$를 정의역으로 갖고 $p$를 치역으로 갖는 함수를 생각하자. 이 함수의 이름이 logistic function이다.

$$
\text{logistic function} = \frac{e^{input}}{1+e^{input}}
$$

이 함수의 그래프는 다음과 같다
![logistic function](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1280px-Logistic-curve.svg.png)

[출처:위키피디아](https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Logistic-curve.svg/1280px-Logistic-curve.svg.png)

이 함수를 피팅해서 쓰는건 생각보다 적절할 것 같다. decision boundary 근방에서 가장 확률이 크게 변하고, 나머지 구간에선 천천히 완만해지고.. 우리의 직관(prior)과 부합한다.

이때 input을 기존의 linear regression에서 썼던 $\theta X$로 주면 되겠다. 다만 bias term까지 넣기 위해서, 계수차항까지 넣어야 한다.

**즉 Logistic Regression은 이 함수를 통해 decision boundary를 fitting하는 작업이다.**

이제 Logistic Regression을 통해서 Y를 추측해보자. 여기서 Y가 확률로서 주어져있으므로, 이것은 yes or no question이다. 따라서 이를 Bernoulli function으로 표시할 수 있다,

$$
P(Y|X)  = \mu(x)^y(1-\mu(x))^{1-y}
$$

이를 MLE 방식으로 추정한다면 다음과 같이 수식을 전개할 수 있을텐데,

$$
\hat{\theta} = argmax_\theta P(D|\theta)
$$

현재 문제는 multiple X를 가정하고 문제를 푸는 상황이기 때문에 조금 식을 수정해서 문제를 풀어야 한다.

$$
\hat{\theta} = argmax_\theta P(D|\theta) \\= argmax_\theta \Pi_{1\leq i \leq N } P(Y_i|X_i;\theta)= argmax_\theta \log\sum_{1\leq i \leq N } P(Y_i|X_i;\theta)
$$

따라서 이때

$$
P(Y_i|X_i;\theta)  = \mu(x_i)^{y_i}(1-\mu(x_i)))^{1-y_i}
$$

이 식을 추가하여 위 식을 정리하면

$$
log(P(Y_i|X_i;\theta)) = Y_iX_i\theta + \log(1+e^{X_i\theta})
$$

또한 이 식의 미분꼴은

$$
\frac{\partial}{\partial\theta_j}\{\sum_{1\leq i \leq N} Y_iX_i\theta + \log(1+e^{X_i\theta}) \} =\\ \sum_{1\leq i \leq N} X_{i,j}(Y_{i}-P(Y_i = 1|X_i;\theta))
$$

이것에 대해서 argmax를 취해주려면 MLE에서 했듯이 위 값을 미분한 후 0이 되는 값을 찾아서 그 값을 $\theta$로 하면 되는데, 실제로 이것을 구해보면 closed form으로 나오지 않기 때문에 위와 같이 하기 어렵다. 그러니까 그냥 무식하게 gradient decent를 활용하는 건 어떨까?

## Gradient decent

가볍게 설명하도록 하겠다.

$$
f(x+h) \sim f(x) + h*f^`(x)
$$

이므로, 위 아이디어를 활용하여 f(x)의 local minimum을 찾자는 이야기이다. 이걸로 $\theta$를 추정한다면, iterative method를 활용하여

$$
\theta_j^{t+1} \leftarrow \theta_j^{t} + h\frac{\partial f(\theta^t)}{\partial \theta^t_j}
$$

다음과 같이 구할 수 있겠다.

## Gaussian Naive Bayes

놀랍게도, 베이지안 방식으로 구한 Gaussian Naive Bayes는 결국 logistic regression의 형태와 똑같다.

잠깐 Naive Bayes에 대해서 Recall하자면, 나이브 베이즈의 식은 다음과 같다.

$$
f_{NB}(x) = argmax_{Y=y}P(Y=y)\Pi_{1\leq i\leq d}P(X_i = x_i | Y= y) (1)
$$

즉 주어진 데이터 $X$에 대해서 $Y$를 최대한 잘 추정해야 하는 optimizer문제에서, 이를 베이지안으로 바꾸어 $P(Y|X)$ 가 아닌 $P(Y)P(X|Y)$ 를 maximize 하자는 것이다.

기존의 Naive Bayes에선 linear한 MAL을 사용하여 이를 측정했는데, 이를 가우시안으로 측정한다면 다음과 같이 될 것이다.

![ㅈㅈ](/images/2020-12-17-13-06-00.png)

conjugate 가정을 위해서 $Y$의 분포도 gaussian이라고 놓았다.

위 식을 (1)에 대입하면 다음과 같이 나온다.

![가우시안](/images/가우시안.png)

이제 위 식을 로지스틱 함수의 꼴로 만들어 보자. 우선

![hi](/images/2020-12-17-13-08-10.png)

다음과 같은 꼴로 바꿔준 다음, 여기다가 가우시안 가정을 추가해 보자.

![ㅈㅈ](/images/2020-12-17-13-09-14.png)

여기서 위 식을 logistic 꼴로 만드려면, $X$를 선형으로 만들어주어야 한다. 하지만 위 식엔 $X^2$가 존재한다. 이 항을 어떻게 없앨 수 있을까?

등분산성을 가정하면 된다.

![naive bayes](/images/2020-12-17-13-16-07.png)

이렇게 되면 이 식은 결국 logistic regression과 꼴이 같아진다.

## Logistic Regression과 Gaussian Naive Bayes의 차이

여기서 위 두 식이 꼴은 같지만, 거기까지 유도하는 데에 들어간 가정들은 사뭇 다르다는 것을 유념해야 한다.

가우시안 naive bayes의 경우 많은 가정들을 하였다.

1. 모든 변수들이 독립이라는것을 가정하였다. (Naive)
2. 등분산성을 가정하였다.
3. prior 분포를 가정하였다.(Bayes)
4. Y의 베르누이 분포가정을 하였다.

또한 위 두 식은 형태는 같지만 추정하는 파라미터의 개수가 다르다.

Gaussian Naive Bayes는

$4d$(각 Attribut의 분산과 평균을 알아야 하므로) + 1 ($\pi_1,\pi_2$ 둘중의 하나는 알아야 한다) 개의 파라미터가 필요하지만,

Logistic regression은 그냥 $X\theta$이니까 bias term까지 합쳐서 d+1개면 된다.

## Generative - Discriminative Pair

$P(Y|X)$를 어떻게 Tackle할것인가를 묻는 것이다.

Generative Model은 이를 베이지안 방식으로 접근해 $P(Y|X) = \frac{P(X|Y)P(X)}{P(X)}$로 보겠다는 것이고, 당연히 이 경우엔 분포를 생성해서 그 분포를 보아야 한다

Discriminative는 이를 분포로 보지 않고, 문자 그대로 $P(Y|X)$를 풀겠다는 것이다.

## 질문

Logistic Regression은 MLE, Naive Bayes는 MAP로 추정한다고 생각할 수 있음. 그런데 왜 굳이 다 MLE혹은 MAP로만 추정하려고 하는가? 덧셈과 곱셈의 차이?