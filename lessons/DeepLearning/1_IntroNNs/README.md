## Neural Networks

__softmax__: a function that convert scores (logits) to probabilities 
$$
softmax(x) = \frac{\exp(x)}{\sum \exp(x)}
$$
__one-hot encoding__: expanding the labels as big as the number of classes and assign one and zeros to each element. e.g., label $2 -> (0, 0, 1)$

__Maximum Likelihood__: SELF-STUDY

__Cross-Entropy__: (a measure of error)

connection between probabilities (sigmoid or softmax values) and amount of error
$$
CrossEntropy = -\sum_{i=1}^n y_i \log(p_i) + (1-y_i) \log(1-p_i)
$$
Where $y_i$ is the one-hot  encoding of the labels and $p_i$ is the (softmax) probability of a class.

__SSE (Sum of the Squared Errors)__:
$$
E = 1/2 \sum_\mu \sum_j [y_j^\mu - \hat{y}_j^\mu]^2
$$
where $j$ is the output unit and $\mu$ is data points.

## Gradient Descent 

#### An overview of gradient descent optimization algorithms: [link](https://arxiv.org/abs/1609.04747)

Method to minimize objective function $J(\theta)$ ($\theta$ is model parameters) by updating the parameters in the opposite direction of the gradient ($\nabla_\theta J(\theta)$). 

The learning rate $\nu$ determines the size of steps.

#### GD variants:

1. Batch GD: 

   Vanilla GD. update on entire training set. Guarantied converge for convex 

   Cons: Memory and time. Does not allow online update

2. Stochastic GD (SGD)

   Update per sample. Fast. Can be used to learn online.

   Cons: high variance update (fluctuation in objective function/learning curve). Converges only when decreasing the learning rate over time; otherwise, overshoots.

   Tips: Shuffle before every epoch

3. __mini-batch GD (also called SGD for DL!)__

   Updates for every mini-batch of $n$ samples.

   Reduce variance of parameter updates

   Computationally more efficient

   Cons and challenges: 

   * Convergence not guarantied. 
   * Tuning best learning rate (schedules and annealing)
   * Same learning rate for all parameters! 
   * Get trapped in saddle points.

#### GD (SGD) optimization algorithms:

1. Momentum:

   Add a fraction (e.g., 0.9) of the update vector of the past time step to the current update.

   Pros: Getting out of local optima and reduce oscillation.

2. Nesterov accelerated gradient (NAG):

   A look ahead moment (calculating the gradient not w.r.t. to our current parameters but w.r.t. the approximate future position of our parameters)

   This anticipatory update prevents us from going too fast and results in increased responsiveness. (Good for RNNs)

3. Adagrad:

   adapts the learning rate to the parameters. (divide learning rate by root of sum of squares of all past gradients of that parameter )

   low learning rates for parameters with frequently  occurring features.

   high learning rates for parameters with infrequent features. 

   Pros: Good for sparse data. Improves robustness of SGD. No manual learning rate tuning ($\nu =0.01$)

4. Adadelta

   Extension of adagrad which reduce its aggressive, monotonically decreasing learning rate (sum of squares of all past updates). It uses a moving (decaying) average (past $w$ elements) RMS error of gradients.

   No default learning rate is needed.

5. RMSprop

   Very similar to Adadelta (does not use RMS of parameters to keep the units).

   soothing factor ( for moving average) 0.9 

   Default learning rate ($\nu=0.001$)

6. __Adam__

   Adaptive Moment Estimation (Adam). 

   Combination of RMSprop and momentum: A heavy ball with friction (prefers flat minima)

   Uses adaptive learning rates for each parameter. 

   In addition to storing decaying average of squared gradients (like Adadelta and RMSprop), it also keeps a decaying average of gradients itself (like momentum).

7. AdaMax

   Similar to Adam (but uses $l_p$ Norm instead of $l_2$ Norm). Large p values are unstable except $l_\infty$ which is used in AdaMax.

   Good default values $\nu=0.002$, $\beta_1 = 0.9$, and $\beta_2 = 0.999$. 

8. Nadam

   Nadam (Nesterov-accelerated Adaptive Moment Estimation). 

   Combines Adam and NAG (modifies Adam's momentum)

9. AMSGrad

   Motivation: Short-term memory of exponential moving average of the gradient became an obstacle for convergence. (SGD w/ momentum outperform it)

   AMSGrad uses the maximum of past squared gradients rather than the exponential average to update the parameters (non-increasing step size). The rest is similar to Adam.

   ##### Note: Adam is the best overall choice, so far.

####  Parallel and distributed SGD algorithms:

1. Hogwild:

   Only when data is sparse.

2. Downpour SGD

    runs multiple replicas of a model in parallel on subsets of the training data.

   Cons: Convergence is not guarantied

3. Delay-tolerant Algorithms for SGD

   extend AdaGrad to the parallel setting by developing delay-tolerant  algorithms that not only adapt to past gradients, but also to the update delays.

4. __Tensorflow!__

   A computation graph is split into a subgraph for every device and communication takes place using Send/Receive node pairs 

5. Elastic Averaging SGD (EASGD)

   links the parameters of the workers of synchronous SGD with an elastic  force, i.e. a center variable stored by the parameter server. 

   

#### Additional Strategies for optimizing SGD:

1. Shuffling (no order) and Curriculum Learning (with meaningful order)

   Curriculum Learning is used to train LSTMs (sort by difficulty)

2. Batch normalization

   Similar to normalizing the initial values of our parameters by initializing them with zero mean and unit variance, Batch normalization reestablishes these normalizations for every mini-batch and changes are back-propagated through the operation as well. 

   __Note: Batch normalization additionally acts as a regularizer, reducing (and sometimes even eliminating) the need for Dropout. __

3. Early Stopping

   monitor error on a validation set during training and stop (with some  patience) if your validation error does not improve enough. 

4. Gradient noise

   Add Gaussian noise to each gradient update.

   Anneal the variance over time.

---

Initial values for parameters: Normal distribution centered at 0 and the scale of $1/sqrt(n)$ where $n$ is number of features.

## Backpropagation

Sigmoid function has the __vanishing gradient__ problem:

maximum derivative of the sigmoid is 0.25 (error magnitude get reduced by 75% only by one layer and vanishes if you have lots of layers).

---

## Training Techniques for Neural Networks

__Early Stopping__ based on `model complexity graph` to avoid overfitting (and underfitting).

Stop were validation is not improving enough!

__Regularization__ is the technique to penalize large weights to avoid overfitting. $\lambda$ hyper-parameter.

- L1 (Norm) regularization (sum of abs values): Ends up with sparse vectors, _good for feature selection_
- L2 (Norm) regularization (sum of squared values): Ends up maintains all the weights homogeneously small. Not good for sparsity. _good for training models_.

__Dropout__ randomly turn off nodes to increase redundancy and generalization in learning

 #### Additional strategies for training NNs from course

1. _Random restarts_ to avoid local minima
2. use various activation functions to avoid _vanishing gradient_ problem
   1. Hyperbolic tangent 
   2. Rectified linear unit (ReLU)
3. SGD instead of batch GD
4. Learning Rate Decay (annealing)
5. Momentum for GD

---

### Keras

read the website for [getting started](https://keras.io/getting-started/sequential-model-guide/)

---

### Pytorch

[Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py)



