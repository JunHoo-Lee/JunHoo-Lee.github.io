---
layout: post
title: "Baysian Network"

categories: prml
---

확률론의 기초에 대해서 배운다.

확률이랑 상대적인 빈도를 의미한다. 즉 given dataset이 단순히 하나의 라벨에 대한 값밖에 없다면, 그것은 확률이 될수 없다.

Joint Probability가 굉장히 중요하다.

Law of Total Probability. 'summing out', marginalization.

$P(a) = \sum_b P(a,b) = \sum_b P(a|b)P(b)$

로 표시할 수 있다. 이것을 marginalization이라고 부른다.

$P(a|b)P(b) = P(b)$

이기 때문이다. joint를 안다면, 개별 각각에 대해서도 알수 있다. joint는 모르고, conditional은 안다, 라고 하는 경우가 있다.

joint를 알면 - marginal probability를 알고 - 따라서 conditional probability를 계산할 수 있다.

이것을 역으로 하면

$$
P(a,b,c,...,z) = P(a|b,c,...,z)P(b,c,...,z)
$$

다음과 같이 인수분해를 할 수 있다.

$$
P(a,b,c,...,z) = P(a|b,c,...,z)P(b|c,...,z)P(c|...z)...P(z)
$$

joint는 다음과 같은 factorization이 가능하고, chain rule이라고 부른다.

## independence

B에 대한 전제조건이 상관없이, A가 나올 확률은 같다.

### Marginal independence

$$
P(A|B,C) = P(A|C)
$$

같은 것을 말한다.

이제 Baysian Network에 대해서 직접적으로 다루어 보자.

Naive Bayes를 생각해보자. 그때 우리는 계산상의 이득을 위해서 y가 주어졌을 때, $X_i$간의 independence를 가정하였다.

그림으로 표현하자면 다음과 같이 될 것이다.

![Node](/images/2021-01-04-17-56-41.png)

이런 식으로 각 변수들의 관계를 표현하는것이 Baysian Network이다.

## Baysian Network

Network에는 Node와 link가 있다. 따라서 위의 Naive Bayes는 Baysian Network이다.

이것이 되기 위한 구체적인 조건은

1. 사이클이 없어야 한다
2. 방향성이 있는 그래프여야 한다.

$P(X_i|parents(X_i))$

와 같은 식으로 확률을 표시할 수 있다. 

direct influence는 직접적인 child와 parent의 관계를 뜻한다.

Network의 구조는 conditional independence를 가정하고 있다. 관련이 없으면 link로 연결되어있지 않다. (independent하면 link가 없다.)

Baysian network은 몇가지 정형화된 형태가 있다.

1. Node들이 공통의 parent를 가지는 경우.

2. Node들이 Cascading하는 경우 (A를 알면 B를 알고.. 그럼 C를 알고..)

3. Common한 child를 가지고 있는 case. child를 공통으로 가지는 구조. V-Structure. 어떤 정보를 알면 특정 정보의 관계가 생겨버림. 알면 알수록 관계가 생겨버리는 상황.

마지막 3번이 문제된다. 알면 알수록 복잡해지니까!

## Bayes Ball Algorithm

다양한 independent 관계가 baysian network에서 굉장히 유용한데, joint probability를 fatorize할때, independent가 쓰이기 때문이다. 이런 문제의식에서 Bayes Ball Algorithm이 만들어졌다.

Ball을 굴려서 



## 코드 분석

[PRML-LECTURE](http://seslab.kaist.ac.kr/xe2/page_GBex27)

<div align="center">
  <iframe width="560" height="315" src="https://www.youtube.com/embed/OZJoBK2slOA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>
