r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**
1. False. 
in-sample error is the error rate we get on the same data set we used to build our predictor.
the dataset is a subset of the dataset that was not used to build the model, hence it estimates the out of sample error. 

2. False.
for example, given a data with two labels (dogs and cats), we can split the data to two disjoint subsets where each 
subset has samples of a specific label (i.e train set = all samples labeled cats, test set = all samples labeled dogs).  
As each sample is uniquely labeled, those are disjoint subsets. the model would probably label everything as cats unlike
a model based on subsets that consist of samples with label distribution which is equal to the data set distribution. 

3. True. 
This question is ambiguous as the definitions of "test set" and "using" vary in different sources.
To answer clearly, we'll first present our terminology:
In CV we first allocate a subset of the dataset to be test set. We then use cross validation on the *rest of the dataset
(*excluding the test set). At this part we name the new subsets "training set" and "validation set". 
So, in the training process of the CV routine, we repeatedly use different training and validation sets, but not the 
test set. Finally, after the training process we use the test set to estimate our models out os sample error. 

4. False.
Following the logic and terminology presented above, the test set is used to estimate our generalization error.  

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part1_q2 = r"""
**Your answer:**
His approach of adding an L2 regularization term to the loss function is justified as it minimizes chances of 
overfitting model due to overinflated weights. However, as a hyperparameter, the regularization rate \lambda should not
be determined by the test set, but optimized by the training process based on the training and validation set.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
In our model, increasing the values of k does not improve generalization for unseen data. 
In our model, k=1 mostly had the highest accuracy, followed by k=3 that somtimes even slightly surpassed k=1.  
We assume that theres high variability of the samples representations, such that for a test sample, 
it is likely to find a training sample that is very close (low l2 dist), so for k=1 we'll get a more accurate decision.
But, the larger k we use, we get more false labels in our top-k (since the l2 dist between wrong labels with similar 
representation is close to that of a right label with "far" representation), which leads to lower accuracy.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**
1. k-fold CV is better than Training on the entire train-set with various models and selecting the best model w.r.t 
train-set accuracy in the *sense of model's ability to generalize its performance to an unseen data*. In a more 
"extreme" manner, in the CV method we decrease the chances of overfitting compared to the suggested method.    

2. In this case, the suggested method is again increasing the risk of overfitting, just this time it iss overfitting
to the test set. In a sense, thie means our model will underfit to the training set. Usually this will lead
to a "bad" model as the training set is usually bigger and "better distributed" than the test set.   


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

**1:** The selection of $\Delta > 0$ is arbitrary for the SVM loss function because delta the hyper-parameter 
representing the margin for score difference between the true and all other classes can be directly effected by the 
hyper-paramater $\lambda$. The "weights decay"\regularization  $\lambda$ parameter controls the magnitude of the 
weights so as we limit this magnitude we are also limiting the difference in scores between the classes $\Delta$. 
Therefore, we can safely set $\Delta$ arbitrarily and have $\lambda$ compensate. The two hyper-parameters control the 
same tradeoff and controlling the weights magnitude may prove to be more convenient in some cases.      

"""

part3_q2 = r"""
**Your answer:**

**2.1:** When viewing the images we can roughly see the outlines of the digit characters. We can assume the classifier 
is learning the pixels and where they are white or black. Ultimately giving the sample a score based of how close it is 
to each class\digit. The classifier learn the features which are the pixels and gives a specific weight to each one and 
based off how often this pixel was white in the training set. The classification errors may be explained by the several 
factors like, how there are different styles to write each digit, how some digits are similar, how the training samples 
are not all the same size, centered or aligned the same.  For example we can see how the digit 4 was mistaken for 9 due 
to an elongated stroke.

**2.2:** 
The similarities are between the interpretation: 
- The models both regard the same features, in this case the pixels. The model is built around pixel location and color.
 
The differences are:
- The KNN model predicts the classification by comparing to the closest training samples to the sample we are trying to
predict, while in the Linear Classifier model we build a general representation for each of the C class options and
ultimately predict the classification based on which class is most similar and not specific samples. 

-In the Linear classifier we have a much smaller memory stamp due to not having to save the whole training set on the 
memory while predicting future classes. 

"""

part3_q3 = r"""
**Your answer:**

**3.1:** We think that the learning rate we choose is good. We can see how in the first 5 or so epochs the accuracy 
function has a general trend of improvement btu with local falls. This is an important element because we want to avoid 
local minimals. The accuracy graph begins to flattens out after epoch 5 thus showing some sort of convergence to a 
minimum this is important because if we were to choose a learning rate to large we might always "bounce" in and out of 
minimums. On the other hand we can notice that our step if large enough to actually reach a minimum, choosing too small 
of a step would most likely cause the model to not reach adequate results.

**3.2:** Based off the graph we can concur that the model is slightly overfitted to the training set. By examining the 
graph we can notice that the accuracy of the training set is better then that of the validation set by a small margin.

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
**Your answer:**
There are several factors that make an ideal pattern in a residual plot 
1. Obviously, as we want to minimise our error, we would like to see a residual plot with a low loss values/avg.
(closest to zero).
2. In a residual plot where we split the dataset to train and test set, we would like to see a similarly distributed 
residual for the two subsets, so we know our model is able to generalize its performance and it is not overfitting to 
one of the subsets.
3. we would like to see a linear horizontal residual pattern so we know that the models performance is "equally good" 
for all values. Which means, we want to avoid the case where our model is very good (=small residual) for some values, 
yet very bad (=large residual) for other values.  

comparing the final model (after cv) to the one with the top 5 features, we see that the final model is better in all
three metrics described above. Thus, we infer that the final model is a better model for our purpose and under out
metrics. 

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q2 = r"""
**Your answer:**

Adding non-linear features to our model can be seen as projecting our samples from one space to another, 
s.t in the new space we can linearly separate our samples better than previous space enabled us. 

1. Yes, it is still a *linear* regressor, as the final classifier is still a linear function. (We did not change the
hypotheses space)   

2. The question is a bit ambiguous, so to be on the safe side we we'll answer both meanings. 
a. if the question is whether we can apply any non linear function on the features to create new features for our
linear regressor, than the answer is yes. any function that will generate a new feature will work (although not
necessarily improve the model).
b. if the question is whether we can "predict" (/imitate) any non-linear function with our linear regressor (using the 
ability to create "non linear" features) than the answer is yes although in many cases it would demand a lot of
calculations or a preliminary knowledge about our data.

3. In the new space, the classifier would be a hyperplane. BUT looking at the original we might get boundaries that are
more complicated than just a hyperplane. for example, we could get a sphere, or multiple hyperplanes that separates
our original space.     

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part4_q3 = r"""
**Your answer:**

1. logspace enables us to examine different magnitudes for our lambda hyperparameter within a shorter time than the 
linspace wouldve provided us. This enables us to find a better value faster. 

2. The cv process has a k-folds iterations. denoting A as the number of values for lambda and B as the number of values 
for the degree, we get that the model was fitted (k_folds)(A)(B) times.
In this specific call, we have k_folds=3, A=3, B=20, so we get 180 times.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============
