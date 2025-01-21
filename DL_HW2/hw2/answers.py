r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**

**1.A.** The Jacobian matrix $\pderiv{\mat{Y}}{\mat{X}}$ will be of shape $(64 \times 512 \times 64 \times 1024)$

**1.B.** Yes. Y will be of shape $(64 \times 512)$ then each value will be differentiated with respect to each value in X which is of shape
 $(64 \times 1024)$. Each one of the 64 samples of Y will be differentiated with respect to one of the 64 samples in X.
 So if we consider the Jacobian matrix elements where the 1st and 3rd dimensions are not equal i.e. we are differentiating one sample with respect 
 to a different sample, then we will get zero since the different samples have no effect on the partial derivates of one another. 

**1.C.** No. We can represent $\delta\mat{X}$ in the following way $ \pderiv{L}{\mat{Y}}\pderiv{\mat{Y}}{\mat{X}} = \pderiv{L}{\mat{X}}
 = \delta\mat{X} $. Therefore if we can calculate $\pderiv{\mat{Y}}{\mat{X_{i}}}$ we calculate $\delta\mat{X}$ without materialzing the Jacobian.
 As noted above if we consider each individual element of $\pderiv{\mat{Y}}{\mat{X}}$ in the locations $(i,j,m,n)$ where $1 \le i=m \le 64$ we get 
 a non-zero value. As shown in 1.b we know that the partial derivative of $\pderiv{\mat{Y_i}}{\mat{X_{j}}}$ is zero when $i \neq j$ , using the 
 chain rule on $\pderiv{\mat{L}}{\mat{X}}$ we can split the derivative by samples. Resulting in $\pderiv{\mat{L}}{\mat{Y}} \mat{W} = 
 \delta\mat{Y}W = \delta\mat{X}$ Therefoe we do not neet to materialize the Jacobian.

 **2.A.** The Jacobian matrix $\pderiv{\mat{Y}}{\mat{W}}$ will be of shape $(64 \times 512 \times 512 \times 1024)$

 **2.B.** Yes similarly to before if we differentiate between two different features we will have a zero gradient. So all elements that the indices 
 of the 2nd and 3rd dimensions are not equal will have a zero value.

 **2.C.** No. Similarly to before, we can respent $\delta\mat{W}$ by spliting it into partial derivatives per each single weight $w_{i}$. Applying 
 the chain rule and summing over all the partial derivatives of $y_{i}$ with respect to $w_{j}$ we get zeros values in when considering different
 features (i.e. $i \neq j$) which leads to $\delta\mat{Y}X = \delta\mat{W}$ and therefore do not need to materalize the jacobian. 

"""

part1_q2 = r"""
**Your answer:**
**2**Technically **no** but in a practical sense (when taking in to consideration computational time) **yes**. As we saw in class we can directly 
calculate gradients of each layer and by the chain rule calculate the gradient of the model to implement descent-based optimization. However, due 
the the extremely demanding computational time and the fact that some function are not differentiable we must use the back-propogation method 
(or something similar to it) to compute a gradient for the model for use in decent-based optimization.  
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======

    wstd, lr, reg = 0.1, 0.05, 0.0005

    # raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======

    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0.1,
        0.025,
        0.015,
        0.0003,
        0.001,
    )

    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    
    wstd, lr, = (
        0.1,
        0.001,
    )

    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

**1.1** - We set the hyper params such that with no dropout the model is overfitted. We can see that by the high training accuracy and the low test 
accuracy. With dropout we can see that with the high dropout rate of 0.8 the model does not reach good results in both the test and training 
batches. With the moderate dropout of 0.4 we can see that the model does not reach the same results in training like without dropout, as expected 
due to the stochastic element of dropout. However during the test we acheive substantially better results with dropout than without dropout, this 
is also expected since the dropout compensates for the tendancy of overfitting by using the probabalistic approach of droping nodes at random.  

**1.2** - With to large of a dropout the model was unable to sufficiently optimize itself during training and thus we can see the non-benificial
results in the testing portion. With the dropout set to 0.4 the model improves substantially its test accuracy. Often the conventional dropout 
rate is 0.5 so we would expect the 0.4 to better improve our results than the 0.8 dropout 

"""

part2_q2 = r"""
**Your answer:**

**2** - Yes it is indeed possible. Loss and accuracy are not always inversely correlated. 
For example lets examine a binary classification, let sample X with true labels of [1,0] be classified as follow [0.9,0.1] the cross entropy will 
be quite low because the classification is close to true labels. Let us also look at the same X and labels but classified as [0.6,0.4], the c.e.
will be quite high this time. If the model for some reason (maybe overfitting, fluctuations due to dropout or mini batching) changes the 
classification from the first to the second we would have the same accuracy but a higher Loss. Due to this type of behaviour and assuming that
the model is susceptable to noise, we can infer a situation where classification values are getting worse but in some samples the labeling has 
changed so that the accuracy increased. Thus giving us a increase in both loss and accuracy

"""

part2_q3 = r"""
**Your answer:**

**3.1** - Gradient descent (GD) is a method of optimizing a model with the use of a loss function and the gradient of the model. We tune the model 
such that with each step it updates its parameters/weights with regard to the direct of the gradient. We aim for a minima, optimally the global 
minima but usually we find a local one (which has been shown to be similar in value in optimizing as the global). Back-propogation (BP) is a 
method to compute the gradients of a nueral network. The gradients are propgated from layer to layer from the last layer (back) of the model and
this method allows us to calculate the gradient with use of the chain rule in a manner more simple then that of manually doing so. 

**3.2** - GD operates in such that it will take a step in the steepest direction possible which is the shortest to the minima. This calculation is
achieved by calculating the gradient of the entire training data set. While SGD only considers a one sample or a "mini-batch" of the training data 
set for each calculation, therefor its step may not be in the steepest direction. SGD offers several advantages like needing less computational 
power for each calculation, or by adding stochasticity to the optimization sometime the model can exit local minimas that would prove to be less 
accurate for the model. 

**3.3** 

-SGD converges faster on large datasets due to the simpler calculation and more frequent updates to parameters. 

-The stochasticity can help "dodge" local minimas by stepping in an non optimal direction. 

-Requires less memory since only one or a minibatch of data is being processed each time. 

**4.A** - Yes it would produce an equivalent gradient. The gradient *operator* is linear there fore if we divide the data into disjoint batches we 
can utilize the addative property of the operator to seperatly calculate the partial gradients and the sum them to represent the gradient of the 
total loss and not just of the particular batch. 

**4.B** - The calculation of the gradient would be done through back-propogation. This requires us to cache previous gradients during the forward 
pass in order to calculate the final gradient. This acumulation of data on the memory while running all the forward passes before the single 
backward pass is likely to be what caused the memory error.

"""


# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    n_layers= 3
    hidden_dims= 7
    activation= "tanh"
    out_activation= "none"
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()
    lr, weight_decay, momentum = 0.015, 0.015, 0.92
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
* $ \textbf{some defenitions to clearify our answer, can be skipped :)}$
* Approximation error is the "distance" between the real function we are tryin to imetate - $f^*_D$, and the "optimal" function in our hypothesis 
space - $h^*_D$. 
* Generalization error is the "distance" between the optimal function in our hypothesis space $h^*_D$ and the optimal function we can get 
based on a given training set, $h^*_S$. 
* Optimization error is the "distance" between the optimal function we can get based on a given training set $h^*_S$ to the actual
function our model has returned, $\bar{h}$
* estimation error is the combined Generaliztion and optimization error (or: overall error minus the approx error)

As we can see in the plots, after some epochs we already achive the optimal accuray/loss values which indicates our estimation error is
low. Since the estimation error is low, bur the overall model is not accurate, we infer that the aproximation error is high. 
so, to sum up:
Approximation error - probably high
Generalization error - probaby low
Optimization error - probably low

"""

part3_q2 = r"""
**Your answer:**
* FNR = False Negative Rate
* FPR = False Positive Rate
* Based on our data, the thershold selecting method returns a 0.46 threshold. 
Which means that a sample with a probability for  for classifying the data as positive. 
This means that in an uniformly spread data, we are overall more likely to classify a sample as positive rather then negative,
which than means we're more likely to get a higher FPR.
"""

part3_q3 = r"""
**Your answer:**
When deciding over our threshold, we should compare the impacts of FNR and the impacts of FNR. 

1. In the case where the disease has a non-lethal symptoms that immediately confirm the diagnosis and can then be treated,
the Impact of getting a False Negative result is minimal, While getting a False Positive results means more expansive and risky
examinations. Thus, we would prefer a threshold that minimizes FPR over FNR. In other words, we would prefer a model with $\textbf{higher
threshold, unlike the one chosen by our model}$. 

2. In the case of a disease with no clear symptoms and high mortality rate, getting a False Negative result is very harmful
while False Positive result is comparetivly less dangerous. Thus, we would prioritize a $\textbf{lower threshold to decrease FNR. 
which is closer to our models threshold}$. 
"""


part3_q4 = r"""
**Your answer:**
1. For a fixed depth and varied width, we observe two patterns:
* The general trend is that $\textbf{higher width leads to higher accuracy}$, Which means that more neurons improve the model.
In terms of the decision boundries, it seems as if the boundries get more sensative and sufiticated s.t there are more
curves. 
* Once incresing the width to a certein level. the accuracy $\textbf{improvement}$ is getting $\textbf{smaller}$. Hence 
the impact of adding more neurons to achive higher accuracy costs a lot in terms of computations/time.

2. In this case, it seems like the depth has a minor inconsistent effect on the accuracy. somteimes increasing the depth
increases the accuracy, sometimes it decreeses and sometimes only a very marginal difference is observed.

3. The deeper model (depth=4, width=8) has a slightly better performence (higher acuuracy) than the wider 
(depth=1, wisth=32). Although both expiriements have the same total number of parameters,
each one has different number of "calculations". In the wider model each feature goes thru 32 parameters, while in the 
deeper model each feature travels thru $8*8*8*8$ (>>32) parameters.

4. Our threshold selection method is based on the validation set and not the training set. This way we can increase the 
models ability to generlize it preformence, as the hyprparameter is set by smaples the model didnt train on. 
Accordingly, this leads to an improvement on the test set.

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    loss_fn = torch.nn.CrossEntropyLoss()  
    lr, weight_decay, momentum =  0.015, 0.01, 0.8
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**

1. Under the assumptions of 1 kernel and no bias for each layer, we get:
* For the regular block: $1*(64*3^2) + 1*(256*3^2) = 2880$ parameters to learn. 
* For the bottleneck block: $1*(64*1^2) + 1*(256*1^2) + 1*(256*1^2) = 896$ parameters to learn.
* (If we missunderstood the quetion and there are several kernels in each layer, then multiply the i-th arg in K_i)

2. For each kernel, the number of FLOPs is calculated by: Kernel_size*output_size. Breaking it down we would get
$ C_in*F^2*H_out*W_out*_C_out $. Hence, qualitatively, the ratio between both blocks would be close to the parameters ratio
which is ~1:4.

3. Were not sure we fully understood the question (we also Looked it up in the Piazza, Ahatsapp & Google:),
so, to answer we'll first present our understanding of the question:
* Ability to combine the input spatially (within feature maps) is the ability to detect different features/patterns in 
a single given featuremap/input.
* Ability to combine the input across feature maps is the ability to detect features/patterns thru several different 
inputs.

Given those definitions, we can say that the  $\textbf{BottleNeck's Block Spatial ability}$ is probaly  
$\textbf{lower then the Regular Blocks}$ as the BottleNeck block reduces the dimensions of the data and 
by that might lose certain information from.  
We belive that the  $\textbf{BottleNeck's compression does not have a direct impact on the ability to combine the input across
feature maps}$. However it is possible that the impact to the Spatial Ability leads to a  $\textbf{secondary impact}$
on the Across one.
"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**

**5.1.1** - 

When K = 32 the optimal number of layers appears to be 2 since it reached the best accuracy. L4 reached better training results but did worse on
the testing data, we can infer from this that it might have overtrained. 

When K = 64, L = 2 is also optimal. Like above we see that L = 4 reached better training acc but wors test acc. By the curves of the graphs we can
assume that once again L=4 was overfit. 

Both in the 32 and 64 filters per layer experiments we can see that when the layers were 8 and above the models failed to train.

It is evident in this experiment that more layers does not necessarly produce better results.

**5.1.2** - 
First the most prominent result is that we see that the models with 8 and 16 layers did not succeed in training and building viable 
predictions. The cause of this is most likely the vanishing/exploding gradient effect which is a common issue in deep layered NNs. This
can be resolved by initializing the weights of the nueral network more accurately by choosing a different variance (2/n is often used to relu). We
can also use a different acitvation function specifically nonsaturated ones like relu, use batch normalization, or to "clip" the gradients by 
regulating their size durign BP. 


"""

part5_q2 = r"""
**Your answer:**

**5.2** -

In this experiment we cna note that the number of filters has different effect on models depending on how many layers they have.
Let's review the results by number of layers per block L. 

L = 2: Produced the biggest variability in test results here we can clearly see how more filters per block damaged the results. This may be due to 
overfitting. When K=32 The model reached a test accuracy close to 67 percent.  

L = 4: For this number of layers we see some variabiity in test results, less than 5 percent. Since 128 layers per block reached the best results 
and 64 the worst, it is hard to infer why this happened but it seems that all three models overfit. 

L = 8: We can see that for this amount of layers we have once again much variability, almost a 10 percent difference. We should note however that
in consistancy with previous inferences we can see that we achieved better results in L=4 showing that more layers does not necessarily give better
results. We can also see that with 64 filters per layer we achieved optimal results. 

When comparing between the results we can see that in both cases changing the filters per layer of number of layers per block can affect the 
accuracy of the models profoundly. How ever we see that in the first experiment we only saw a difference of about 3 percents in the models with 
fixed K and changing L. In the second experiment in some cases we saw a 10 percent difference. It should be noted that in both experiments there
appears to be a sweet spot where there is little effect of these changes. We think that we should aim for this spot of balance between L and K when
designing a model. It should be noted that many of these models experienced overfitting, which makes deducing conclusions on this data more 
difficult.

"""

part5_q3 = r"""
**Your answer:**

**5.3** -

We can see that in this case L = 4 failed. This is not suprising though. We saw in previous models how the the models failed to train for 8 layers
with high K. In this format we are in effect creating twice the depth that is stated in L due to the K list argument. We can see that for L3 that
the model reached best or close to best accuracy as all other models at around 72 percent accuracy. How ever it reached similar results to L = 4
with K = 64 from exp 1. It looks like there is some extent of overfitting here, since the loss begins to increase at some point and the accuracy
seeems to have a trend of decrease, however since the accuracy trend is minimal it is not so clear cut of an observation.

"""

part5_q4 = r"""
**Your answer:**

**5.4** - 
First we can see that this resnet model gave us improved accuracy results compared to the CNN models. With L = 4 and K = [64, 128, 256] reaching 
close to 80 percent accuracy. 
In comparison to the previous experiments we can notice that this model is able to succesfully train deeper models and also yeild better results. 
In 1.3 we saw how the models with multiple filters per layers sizes failed quite early and in 1.1 we saw how in general the deeper models failed 
to be trained. We can infer that the residual blocks of the res net. In addition we can once again see how adding more layers does not neccesarily 
lead to better model results. 

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**

**1.1** - The model did not detect the images very well. In the first image we can see that it classified the left-most dolphin as a bird and the 
others as a person and bird. In the second image we can see that the left dog was not detected and the middle dog got a low confidence score.

**1.2** - In the first photograph we failed the model from the begging since there is no dolphin classification. To resolve this we can train the
model to classify dolphins aswell. An additional method that would help the results could be to brighten the image. The dolphins are very dark 
making their boarders less pronounced. This may be why the image over layed the two bounding boxes very closely and was not able to realize that
one is occluded by the other. Additional we can suspect that the classifications to birds was partially because of the dolphins appearing on the
background of the sky and sun, when adding the dolphins to the training of the model we should remeber to add varied images. 

The second image appears that due to the density of objects in the image the model did not recognize the left dog. This could be because of 
occlusion. This may be resolved by cropping the image differently or altering the parameters for the bounding boxes. The yolo algorthim does an
initial division of the the image to boxes to look for centers and possible the sizes here were not fitting for the image, possible altering
the size of the initial boxes could help. 

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**

**3** - Poorly. We intentionally uploaded challenging images as instructed. Let's explain how each image faulted the model. 

- The three images of chairs from above are a great example of model bias. We suspected that the model relies heavl on the fact that a chair
has legs and on the shape of the back rest. Thus we uploaded images of the chairs in angles that did not show those features. An ordinary person
can easily detect these objects as chair (or at least not as a toilet or vase like the model). It is likely that from the angle the chairs 
appeared to the model as toilets due to toilets having similar features to that of a chair but possibly more oftenly photographed from above. In
addition it is possible that the image of the chairs in the circle threw off the model due to the symetry of the image and the very specific order
of the chairs. It may have "lit up" different features that are not connected with the chair classification and that the image was taken as a whole
maybe even a flower. However after cropping the image for the bonus portion we can see that the chair was not classified, it may be that the image
is too small or pixelated or maybe since it was taken from the internet it has some introduced noise.

- The next set of images we can inspect are the of the laptop on the table with the bottle occluded behind it, giving us a good example of 
occlusion. We uploaded the image where the model properly classified the bottle behind the laptop to illustrate that possible the cap is a 
feature the model detects, since when the cap is showing the classification worked and when the bottom of the bottle showed the model classified
the bottle as a cup. Once again to a human eye it is obviously not a cup and it is a bottle but the occlusion of specific features faulted the 
model.

- Lastely we can look at the blurry image. In the image the person is properly classified but the bicycle and the person riding it are not. We 
can suspect that it was not classified specifically due to the blurriness of the bicycle and cyclist which are more blurred than the person 
walking. 

"""

part6_bonus = r"""
**Your answer:**

**Bonus** - We attempted to fix the images by cropping them. Unfortnately this did not help any of the images. In an addictional attemped we 
sharpened the blurred image in photo shop which also did not help. In general we think that these images are quite extremeties and that it would
be difficult to improve their classification with only altering the images. With altering the model itself we could probably reach better results.
Possibly by adding images to learn of the bottom of the bottle, blurred images of bicycles, or chairs from additional angles.

"""