---
layout: post
title: "Feature-map-level Online Adversarial Knowledge Distillation"
categories: mipal
---


awesome *** 

hilten, fitnet, attention transfer, 

## knowledge prerequisite

1. Adversarial training Framework
2. KL Divergence

## Abstract

Class probability 뿐만 아니라 feature map 까지 transfer에 포함하는데, 그 방법이 adversarial training framework이다.

## Introduction

network의 경량화 필요 $\rightarrow$ Knowledge distillation
i.e. teacher의 score를 mimic하는 방식으로 경량화를 꾀함

최근엔 pre-trained teacher 대신 선생과 학생이 같이 network를 train시키느 peer-teachering manner도 가능하다는 것도 증명되었음. 이를 online distillation이라 부른다.

## Method

## Questions

1. 단순히 score를 mimic하는것만으로 어떻게 경량화를 이끌 수 있는가? 학습이 더 빨리 될수 있다고는 생각하지만...
2. online distillation
