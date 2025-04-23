# Welcome to the codebase for Online learning inspired Supervisory Control
## The codebase is set up like this which is used in our work:
- PD_quadtotor.py is the code for PD controlled quadrotor. 
- direct_shootinf.py is the NLP/Direct collocation based on Casidi
- PMP_pytorch_noref.py is the codebase for our framework using PMP.

## Additional Codebases not considered in our work but has been used as a developing base
- PMP_pytorch_ref.py is with a referene line so that we train the model and understand the behavior
- PMP_pytorch_dist.py is a codebase to test with disturbances and obstructions. This still needs work
- PMP_normalised.py is a work in progress to scale and normalise variables and parameters for low cost and better convergence.
-PMP_feedback net is a RL based neural network which learns the different path and tries to ignore the local minima. work in progress and issues with data structuring.
- Quadrotor_test.py is a test file and can be ignored