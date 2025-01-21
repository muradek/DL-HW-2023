r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

def part1_rnn_hyperparams():
    hypers = dict(
        batch_size=0,
        seq_len=0,
        h_dim=0,
        n_layers=0,
        dropout=0,
        learn_rate=0.0,
        lr_sched_factor=0.0,
        lr_sched_patience=0,
    )
    # TODO: Set the hyperparameters to train the model.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 512
    hypers['seq_len'] = 64
    hypers['h_dim'] = 1024
    hypers['n_layers'] = 3
    hypers['dropout'] = 0.4
    hypers['learn_rate'] = 0.001
    hypers['lr_sched_factor'] = 0.1
    hypers['lr_sched_patience'] = 5

    # ========================
    return hypers


def part1_generation_params():
    start_seq = ""
    temperature = 0.0001
    # TODO: Tweak the parameters to generate a literary masterpiece.
    # ====== YOUR CODE: ======
    start_seq = "by William Shakespeare\n\nACT I. SCENE 1."
    temperature = .05   
    # ========================
    return start_seq, temperature


part1_q1 = r"""
**Q1: We split the corpus into sequences to partition the process of training into 
smaller segments which greatly simplifies computation memory and complexity. Training
the model over a the whole corpus would most likely cause vanishing/exploding gradient 
issues that even the gated method coudln't overcome**
"""

part1_q2 = r"""
**Q2: We implemented the model such that we use the state (hidden state) of the previous
batch upon training the next batch. Since we implemented the batches in a continous manner 
the model can use old information to produce outputs dependant on previous sequences. 
This allows the model to create outputs that seem to have longer memory than the sequence length.**
"""

part1_q3 = r"""
**Q3: Generating text is a task with dependance on order, meaning that we need to account for the order in 
the sequence of outputs the model provides. The learning itself has to take order in account other wise we
would get random text generation and not actualy words and structure of the text that we would like to learn. 
That is why we implmented the model and arranged the data sets to hold continous batches, if we were to 
shuffle them we would loose the information stored in the order between the batches.**
"""

part1_q4 = r"""
**Q4.1: We lower the temperature to cause the probability distribution of the next character to generate to 
be less uniform. As mentioned above a low T will cause the dist. to be less uniform. Using T=1.0 will produce 
the regular softmax function and as stated above this can lead to uniform dist. when scores are similar 

Q4.2: When the temperature is very high we can see form the graph that we get a uniform dist. Thus giving us 
text with almost random choice of characters.

Q4.3: When the temperature is very low we get a very un-uniform distribution giving the highest scoring character
much more weight, this leads to very repetitive text (same sentences over and over), this is due to the model loosing 
the randomness given to the dist. by the temp.**
"""
# ==============


# ==============
# Part 2 answers

PART2_CUSTOM_DATA_URL = None


def part2_vae_hyperparams():
    hypers = dict(
        batch_size=0, h_dim=0, z_dim=0, x_sigma2=0, learn_rate=0.0, betas=(0.0, 0.0),
    )
    # TODO: Tweak the hyperparameters to generate a former president.
    # ====== YOUR CODE: ======
    hypers['batch_size'] = 64
    hypers['h_dim'] = 128
    hypers['z_dim'] = 32
    hypers['x_sigma2'] = 0.0009
    hypers['learn_rate'] = 0.0005
    hypers['betas'] = 0.87, 0.9995
    # ========================
    return hypers


part2_q1 = r"""
**Your answer:**
* in terms of optimization:
${\sigma^2}$ is a regularization value that determines the ratio between the reconstruction loss and KL divergence loss in
the loss function used to optimize the model. Lower ${\sigma^2}$ values make the model minimize the reconsruction loss,
while higher values grants a higher focus to the KL divergence loss. 

* Statistical and empirical meaning:
The ${\sigma^2}$ hyperparameter is the std of the parametric likelihood distibution from which the hidden representaition z
(posterior) is sampled. Evidently, lower ${\sigma^2}$ means the reconsructed images will look more similar to the images in 
the training set, as we dont allow a high varriance. Higher variance might generate more creatively, but the output will be
very blury and noisy. 
"""

part2_q2 = r"""
**Your answer:**
1. Reconstruction loss is the part we will usually see in a AE models, and it is aproximating the distance between our 
reconstructed data and the input data (i.e: aims to minimize the effect of encoder${\circ}$decoder on the data). The second
part - KL divergence loss is the part that is unique to the VAE model. It's a statistical measurement of the distance between 
two given distributions. In the VAE model, the KLD represents the distance between the distribution the model is learning 
(by the values mu and and log_sigma_z) and the standard normal distribution. 

2. As we "force" the latent-space distribution to be closer to ~N(0,1), we get a distribution that is more continuous 
and less noisy to represent our latent space.  

3. The benefit of this effect is enabling us statistically sample latent vectors such that the decoder will generate new data,
and this data wont be too noisy (-> would generate better images). 
"""

part2_q3 = r"""
**Your answer:**
P(X) is the evidence distribution, which is the distribution of the data we have (in our case JWB images). In order to 
generate images "similar" to this data (i.e, to generate JWB images) we need our model to have higher likelihood distribution.
Hence we need to maximize the evidence distribution P(X) in the VAE loss function. 
"""

part2_q4 = r"""
**Your answer:**
There are two main reasons to use the log of the variance:
1. By definition, variance should always be positive. By letting the model learning the log of the variance, we "force it"
to produce a valis result as any log would lead to a positive var.
2. Usualy ${\sigma\in[0,1]}$ which is a small number. using its log enables the model to learn a higher range of numbers,
which will lead to both higher accuracy and numerical stability of the model.
"""

# ==============

# ==============
# Part 3 answers




part3_q1 = r"""
**Your answer:**


"""

part3_q2 = r"""
**Your answer:**

"""

part3_q3 = r"""
**Your answer:**

"""

part3_q4 = r"""
**Your answer:**

"""


# ==============
