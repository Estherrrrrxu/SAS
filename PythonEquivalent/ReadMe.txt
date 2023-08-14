

model
  |
  | ---- model_interface.py: This is the script to connect all necessary       
  |             |            information from the rest three scripts.
  |             |
  |             | ---- dataclass Parameter
  |             |                    | ---- input_model
  |             |                    | ---- transition_model
  |             |                    | ---- observation_model
  |             |
  |             | ---- class ModelInterface
  |                                  | ---- input_model                    
  |                                  | ---- update_model
  |                                  | ---- transition_model
  |                                  | ---- observation_model
  |
  | ---- ssm_model.py: Script for running multiple Markov params and making
  |                    inferences.
  |
  | ---- utils_chain.py: Script for each chain running sMC and pMCMC steps
  |
  | ---- your_model.py: Place holder script for your own model. 
  |                     (pass into ModelInterface as customeized_model)

