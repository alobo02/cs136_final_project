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
