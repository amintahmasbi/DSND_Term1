## Support Vector Machines

SVMs attempts to maximize the distance from the linear boundary to the closest points (called the support vectors).

SVMs can be implemented in three different ways:

1. Maximum Margin Classifier
   - linear version for separable data
2. Classification with Inseparable Classes
   - C parameter to tune
3. Kernel Methods
   - nonlinear boundaries
   - Two types: 
     - polynomial: Tune `degree`
     - rbf: Tune `gamma`

#### Maximum Margin Classifier

When your data can be completely separated, the linear version of SVMs attempts to maximize the distance from the linear boundary to the closest points (called the support vectors)

#### Classification with Inseparable Classes

- Cost Function (Error): Classification Error + Margin Error (same as L2 regularization term)

- Margin Error: Max separation/Margin between classes 

- _C parameter_: __C__*classification Error + Margin Error

  Shows our tolerance to false-detection

  - Small C -> Large margin but makes some classification errors
  - Large C -> Well classified points but may have a small margin
  - Ranges between 0 and infinity

#### Kernel methods

Kernels in SVMs allow us the ability to separate data when the boundary between them is nonlinear. 

feature mapping: maps from attributes to features. e.g., $\phi(x)=[x,x^2,x^3]$

- _Polynomial kernel_: include the polynomial combination of attributes (original inputs) ($x, y$) which increases the dimension of the problem but may solve it (adding $x^2, y^2, xy$ features for second degree). 
  - _Note_: The degree of polynomial is a hyper-parameter 
- __RBF Kernel__: Radial Basis Function which puts a normal distribution times a coefficient (figures below) on top of each sample point (feature mapping is a density function) and add them up, so it can separate the classes, and the classification region is the union of these ellipse-like kernels.![RBF Concept](./NotesImages/RBF_Kernel_1.png)

![RBF_Kernel_2](./NotesImages/RBF_Kernel_2.png)

![RBF_3](./NotesImages/RBF_3.png)

- __$\gamma$ Parameter__ : 
  $$
  \gamma = \frac{1}{2\sigma^2}
  $$



where $\sigma$ is the standard deviation of the normal distribution function (RBF kernel) 

_Note_: in higher dimensions the formula for this parameter becomes more complicated but the concept remains the same

![gamma_parameter](./NotesImages/gamma_parameter.png)



#### SVM in sklearn

```python
from sklearn.svm import SVC
model = SVC()
model.fit(x_values, y_values)
```

Hyper-parameters:

- `C`: The C parameter.
- `kernel`: The kernel. The most common ones are 'linear', 'poly', and 'rbf'.
- `degree`: If the kernel is polynomial, this is the maximum degree of the monomials in the kernel.
- `gamma` : If the kernel is rbf, this is the gamma parameter.

