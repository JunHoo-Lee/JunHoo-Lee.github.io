---
layout: post
title: "laboratory ReadMe"
categories: mipal
---

## Maximum-Entropy Adversarial Data Augmentation for Improved Generalization and Robustness

Adverserial data augmentation - 그러니까 사진을 임의로 crop하거나, 색반전 시키는 것 -> robust한 deep neural network를 만듬.

problem : 어떤 heuristics로 해당하는 이미지를 generate 할 것인가?

proposal : generation하는데 필요한 regularlization term을 제시하겠다.

reason : 실제로 해보면 state-of-the-art를 significant margin으로 능가한다.

### Introduction

deep learning, effective하지만, effective하지 않음. -> solution : Adversarial data augmentation

문제: heruistic loss function이 large data shift를 수행하는데엔 좋지 않다. 즉 그런 경우엔 model이 여전히 vulnerable하다.

idea : 정보이론 분야의 IB principle. 모델을 최적의 representation으로 학습하게? 하는데. 그 이유는 irrelevant part of the input variable을(i.e. prediction하는데 필요 없는 variable)을 제거하기 때문이다.

### Background

#### Information Bottleneck Principle

#### Adversarial Data Augmentation





## reference

[PAPER](https://arxiv.org/pdf/2010.08001.pdf)
