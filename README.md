# TOPR-TorchRL

# Env setup

create torchrl environment based on https://github.com/pytorch/rl


# TODOS:

1. Get dataset
  Use SAC to get a bad, medium and good dataset for HalfCheetah. 100k samples?

    -- DONE -- 1mio samples for expert, medium and untrain SAC POLICE [X]

2. Run tests for IQL, CQL, TD3+BC
  Track reward, run-time.

    -- DONE -- Training scripts working for IQL, extension to CQL, TD3+BC can be added quickly [X]

3. Implement TOPR for RL. Check what adaptations need to be done for RL. discounted reward etc maybe ask chatgpt.


4. If successful run on more datasets -> repeat 1&3
5. Paper?

4.2. Can we make it online??
 - add noise to exploraiton action selection.