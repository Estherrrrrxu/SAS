from model.input_generator import InputGenerator
from model.model_interface import ModelInterface
from model.utils_chain import Chain
from model.ssm_model import SSModel
import pandas as pd

# %%
df = pd.read_csv('data/df.csv')

model_interface = ModelInterface(
    df = df,
    customized_model = None,
    theta_init = None,
    config = None,
    num_input_scenarios = 10
)

model = SSModel(
    input_generator = InputGenerator(),
    model_interface = model_interface,
    num_parameter_samples = 15,
    len_parameter_MCMC = 20,
    learning_step = 0.75
)
