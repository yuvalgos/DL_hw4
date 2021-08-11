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

Random signals lesson: 
Note that $ v_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \cdot q_{\pi}(s,a) \cdot p(a|s,\pi)  $

Note that because we take the mean over many trajectories, we hope that due to LLN our optimization will push
$\hat{v}_{\pi}(s)$ to be approximately (the many trajectories implicitly factor in the 
probability of each action given a state) $\mathbb{E}({\hat{q}_{i,t}|s_0 = s,\pi})$

This is more or less $\mathbb{E}({g(\tau)|s_0 = s,\pi})$ which is the definition of $v_{\pi}(s)$.

Thus choosing the estimated q-values as regression targets for our state-values leads us to a valid
approximation of the value function.

"""


part1_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
