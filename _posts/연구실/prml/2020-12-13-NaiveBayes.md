---
layout: post
title: "Naive Bayes"
categories: prml
---



## Optimal Classification

X가 주어졌을 때, y의 값을 추측하는것이 목적.

주어진 데이터를 활용하여 $y$를 추측하는 함수 $f$를 만들 때, 당연히 추측한 $f$들 중 가장 값을 잘 추측하는 $f$를 선택해야 한다. 이를 수학적으로 표현한다면,

$$
f^* = argmin_f P(f(X) = \neq Y)
$$

$$
f^* = argmax_{Y=y}P(Y=y|X=x)
$$

가 될 것이다. 당연하게도, 이렇게 구한 Classificer가 '완벽' 할수는 없다. 이렇게 추정한 값에 대한 리스크 $R(f^*)$가 존재한다. 데이터가 부족해서일 수도 있고, 아니면 Noise 때문일수도 있고...

이 $R(f^*)$를 Baysian Risk라고 부른다.

또한 위에서 구한 $f^*$를 베이지안 방식으로 본다면,

$$
f^* = argmax_{Y=y}P(Y=y|X=x) = f^* = argmax_{Y=y}P(X=x|Y=y)P(Y=y)
$$

가 될 것이다. (why? normalizing constant는 $P(X=x)$ 로, y와 무관한 값이기 때문에 argmax를 취할 때에는 영향을 주지 않는다.)

왼쪽 항은 Conditional density, 뒤쪽 항은 Prior가 될 것이다.

## Decision boundary

Decision boundary란 글자 그대로 우리가 결정을 바꾸는 경계지점을 뜻한다. 그림으로 표현한다면

![예시](/images/2020-12-03-11-18-07.png)

이런 검정색 선이 될 것이다. 기존의 baysian classifier에서 보자면, 이런 선을 바꾸는 경계는 우리가 구한 $P(f(x))$값이 같아지는 지점이 될 것이다. argmax를 취하니까...

### Conditional Problem.

우리가 앞서 본 decision boundary는 다행히도 모두 X값이 한개 아니면 두개였다. 하지만 이를 결정할수 있는 값이 한두개가 아니면 어떻게 될까?

![예제](/images/2020-12-03-11-27-44.png)

다음과 같은 경우를 생각해 보자.

하늘이 좋고, 온도는 따듯하며, 습하고, 바람이 불고, 물이 따뜻하고, 날씨가 계속 바뀌는 날에 대해서 나가놀기 좋은지를 알려면 어떻게 해야 하는가?

이 경우 위의 조건에 대해서 나가놀기 좋을 확률과, 그렇지 않을 확률을 각각 구해서 비교해주어야 한다. 하지만 이 예제만 하더라도, 모든 조건에 대응하여서 알려주기 위해선, 총 $(2^d-1)k$만큼의 계산을 통해서 (여기서 d는 조건의 개수, k는 class의 개수) 모두다 메모리에 저장해주어야 한다. 하지만 이게 말이 되는가? 너무 많다....

이렇게 메모리와 계산량을 많이 먹는 bottleneck은 d이다. 데이터가 d에 대해서 expotentialy 증가하기 때문에 이런 문제가 발생하기 때문이다. k는 선형적이어서 상대적으로 괜찮다. 그렇다면 d를 줄여야 되나? 말도 안된다... (물론 DQN같은 곳에선 줄여주긴 하지만), 열심히 구한 데이터를 우리가 버릴 이유가 뭔가?

다른 방법으로 줄여야 한다. d에 대해 expotentialy 증가한 이유는 우린 조건사이에 dependency가 있다고 생각해서이기 때문이다. 이 때문에 모든 세세한 경우에 대해서 고려해주어야 한다. 따라서 우린 이 문제를 해결하기 위해 단순무식하게, 모든 조건이 independent하다고 가정한다.

이렇게 했을때의 이점은 자명하다. 위의 하늘이 좋고.. 예제를 다시 살펴보자. 이제 저 문제를 풀기 위해선 다음과 같이 계산해주면 된다

$$
argmax_{Y=y} \Pi_{x \in X} P(x|y)P(y)
$$

이젠 계산량이 $(2^d-1)k$에서 $dk$로 줄어들었다.

물론 여기서는 문제를 풀기 위해서 어거지로 설정한 문제들이 많다.

1. 일단 독립가정자체가 말이 안된다. 저렇게 수많은 parameter가 있는데 저걸 다 독립가정한다고?
2. MLE의 경우는 관측된 데이터가 많아야 좀 쓸만한데, 이 경우 특정 조건 하나하나에 대해서는 데이터가 적어서 극단적인 추정을 할 가능성이 높다.
3. MAP도 말이 안되긴 한다. prior를 어떻게 줄것인가? 이게 왜 이런식으로 분포하는지에 대한 이유가 없긴 하다

## 왜 MAL을 사용하는가?

사실 아예 관측되지 않은 데이터에 대해선 아무것도 모르기 때문에 prior를 곱한다는건 큰 의미가 없는 말인것 같다. 그냥 확률 0.5를 공통적으로 줘도 되니까.. 그것보다는 그냥 현실적인 이유같은데, 단순히 이 Naive Bayes Classifier에 MLE를 쓰기 힘들기 때문이다. 앞서도 이야기했지만 데이터셋이 작으면 극단적인 값이 나오기 쉬워진다. 0 또는 1 같은... 이런 극단적인 값이 왜 안좋냐면, 0의 경우 다른 값들을 다 죽여버리기 때문이다. 만약 독립가정을 하고 MLE를 한다고 해보면,

$$
argmax_{Y=y} \Pi_{x \in X} P(y|x)
$$

가 될텐데, 전부다 곱하는 꼴이기 때문에 0이 나오는 경우 우리가 구한 다른 feature의 특성을 전부다 죽여버리기 때문에 하나의 특수한 feature가 다른 값들을 전부다 가려버린다. 이건 직관적으로 당연히 나쁜데다, 실제로 우린 이런 $\Pi$ 형태의 연산을 log를 씌워서 더해버리는거로 처리하기 때문에 $log(0) = -\infty$  가 되어버려서 컴퓨터상에서 계산할수도 없다.

질문 - 선형 - 곡선형 어떤 의미, Naive CLassifier가 선형?? 기하적으로 이해하고 싶다.

## 코드 분석

[CS234-LECTURE](https://www.youtube.com/watch?v=j080VBVGkfQ)

<div align="center">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/j080VBVGkfQ" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
