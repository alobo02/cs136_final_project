# cs136_final_project

## Checkpoint 0:
- Complete

## Checkpoint 1:
- Complete
- Decided to change some of the hypotheses from the submission
    - Specifically, want to make a change to the Prior rather than conduct PCA
    - Original Prior (called the "trivial prior") will be a Gaussian normal, 0 mean, with some precision alpha
    - Upgraded Prior (call the "spike-and-slab prior") will be a GMM, with $\pi_1 = 0.8$ and $\pi_2 = 0.2$
    - Spoke to Ike about this, and he thought it would be an interesting change to explore
        - He wasn't sure how the model will react
        - Didn't think two "upgrades" would be too much work

## Checkpoint 2:
- Developing a class hierarchical structure to develop our algorithm (before upgrade)
- Friday:
    - Nate went ahead and coded the _init_ and the gradient descent algorithm (first-order)
    - Alex needs to go in an make some edits / code up the other methods

## Pre-Checkpoint 3:
- Spoke to Ike during office hours and he reccomended doing the following to help troubleshoot our code:
    - Make your stepsize very small
    - Try running it without the prior to see if it converges
    - Try doing Batch/Mini-batch gradient descent
    - Run without a termination condition to see if it converges after X iterations
    - Try changing the loss function to be the product of the likelihood and prior
    - Run the data with sklearn (without regularization) so that we can compare to our model without a prior
    - Consider running code with autograd in case we have messed up our derivative equations
- General notes:
    - Looks like our sign might be inverted?

## Checkpoint 3:
- Nate worked on section 1. Looks mostly done now
- Alex worked on equations for Upgrade 1.
    - Still need to work on implementation
- Alex aked Nate to help with equations for upgrade 2 since it's taking a bit long
