# dddg
Machine Learning Ducky Labeling and dddg game solver.

![example](https://cdn.ionite.io/img/fIUvgt.jpg)

## Tldr how to use
```shell
pip install dddg
```
> `Usage: dddg [OPTIONS] URL`
```shell
dddg https://cdn.ionite.io/img/Rjj91N.jpg
0 5 11
1 3 10
4 7 9
2 5 9
3 6 7
6 9 10
1 4 6
```

## Introduction

Duck Duck Duck Goose (dddg) is a game whose objective is to
find all "schools" of ducks. A valid school is defined where
each duck can either have the same or different attributes.

![duck_e1](https://cdn.ionite.io/img/Rjj91N.jpg)

- Each card has 4 features
  - Color, Number, Hat, and Accessory
- A valid flight
  - 3 cards where each feature is either all same or all different
  - ![duck2](https://cdn.ionite.io/img/Lgy1TX.jpg)


## Requirements
- torch
- torchvision
- opencv-python
- requests
- webcolors

## Model

Description WIP

![training](https://cdn.ionite.io/img/l8frRh.jpg)
