# 7/21/25
This is a simple 1-variable function approximator that uses backpropagation.

## Log
I coded this up today as an basic example of neural network training. I wanted to make sure I could get some fundamentals down after the difficulties I had with `image-approximator` experiment. Fortunately, this project worked as intended, demonstrating the validity of my backpropagation calculus (*phew*).

This simple project showed me just how long NN training can take, even with just a few neurons. I also got a feel for how training responds to changes in the learning rate and hidden layer sizes.

![Basic curve](https://github.com/tang0226/nnjs2/blob/main/experiments/function-approximation/screenshots/basic-curve.png)
![Overfitted discontinuity](https://github.com/tang0226/nnjs2/blob/main/experiments/function-approximation/screenshots/discontinuity.png)
