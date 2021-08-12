r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""


# ==============
# Part 1 answers


def part1_pg_hyperparams():
    hp = dict(
        batch_size=32, gamma=0.99, beta=0.5, learn_rate=1e-3, eps=1e-8, num_workers=2,
    )
    # TODO: Tweak the hyperparameters if needed.
    #  You can also add new ones if you need them for your model's __init__.
    # ====== YOUR CODE: ======
    hp['batch_size'] = 8
    hp['hidden_layers'] = [128, 128]
    hp['num_workers'] = 2
    # ========================
    return hp


def part1_aac_hyperparams():
    hp = dict(
        batch_size=32,
        gamma=0.99,
        beta=1.0,
        delta=1.0,
        learn_rate=1e-3,
        eps=1e-8,
        num_workers=2,
    )
    # TODO: Tweak the hyperparameters. You can also add new ones if you need
    #   them for your model implementation.
    # ====== YOUR CODE: ======
    hp['learn_rate'] = 1e-4
    hp['batch_size'] = 16
    hp['beta'] = 0.1

    # ========================
    return hp


part1_q1 = r"""
**Your answer:**

Subtracting a baseline in the Policy gradient decreases the size of the weights multiplied by the gradients
of the log-prob of the Policy.
Qualitatively, a baseline is a function when added to an expectation, does not change the expected value,
but at the same time, it can significantly affect the variance.


Example - $\nabla_{\theta} log \pi_{\theta}(\tau) = [0.7, 0.2, 0.1]$ for three different trajectories.
Furthermore the rewards are $r(\tau)$ is [500, 100, 20] (respectively).

We get that $ Var( [0.7*500, 100*0.2, 0.1*20] ) = 38388  $

Let's take the mean reward 206 as a baseline and recalculate.
$ Var( [0.7*(500-206), 0.12(100-206), 0.1*(20-206)] ) ~= 16981 $

In general this is helpful when the scales of the reward for different trajectories changes dramatically causing large variance for the gradients. 

"""


part1_q2 = r"""
**Your answer:**

Note that $ v_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \cdot q_{\pi}(s,a) $

Note that because we take the mean over many trajectories, we hope that due to the law
of large numbers our optimization will push
$\hat{v}_{\pi}(s)$ to be approximately (the many trajectories implicitly factor in the 
probability of each action given a state) $\mathbb{E}({\hat{q}_{i,t}|s_0 = s,\pi})$

Since:
$$
q-values= $\hat{q}_{i,t} = \sum_{t'\geq t} \gamma^{t'}r_{i,t'+1}
$$
$$
g_t(\tau) = r_{t+1}+\gamma r_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k r_{t+1+k}
$$

This is more or less $\mathbb{E}({g(\tau)|s_0 = s,\pi})$ which is the definition of $v_{\pi}(s)$.

Thus choosing the estimated q-values as regression targets for our state-values leads us to a valid
approximation of the value function.

"""


part1_q3 = r"""
**1**

In the mean_reward graph, we can see a standard learning curve. the reward the agent manages 
to get raises through out the training.

In the policy loss graph, we can see similar convergence for the non-baseline experiences, but
a constant curve at around zero for the baseline methods. This happens because the baseline 
term we subtract from the weight of each sample normalizes the loss to be around zero, while
only the base line changes through the training as we see in the base line graph. 
We can see that for the base line methods, the baseline, which represents the average q value
raises which means the model improves.

Finally, in the entropy loss graph we can see that the negative entropy raises, which means
that the entropy drops until some point in the training, then it stays quite constant, which means
that actually the model chooses less diverse policies. That probably happens because the policy
loss takes more weight of the total loss, so the model chooses better policy instead of a more
diverse one. The entropy loss is only a restraining factor to make the model do more exploration.

**2**
The ACC performed as good as the CPG in the graphs, but that was before our final hyperparameters
tuning, which made it converge a little slower, but get better results. 
In addition, ACC was a little bit less stable, it diverged after it got some good results but
then converged again. that happened probably because we estimated the baseline with a neural
network which added some instability.  

"""
