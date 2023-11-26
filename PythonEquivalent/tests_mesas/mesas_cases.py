#%%
sas_specs_invariant_q_u_et_u={

    "Q":{
        "Q SAS function":{
            "func": 'kumaraswamy',
            "args": {
                "a": 1.,
                "b": 1.,
                "loc": 0.0,
                "scale": 3533.,
            }
        }
    },

    "ET":{
        "ET SAS function":{
            "func": 'kumaraswamy',
            "args": {
                "a": 1.,
                "b": 1.,
                "loc": 0.0,
                "scale": 2267.,
            }
        }
    }
}




sas_specs_invariant_q_g_et_u={
    
    "Q":{
        "Q SAS function":{
            "func": "gamma",
            "args": {
                "loc": 0.0,
                "scale": 8215.,
                "a": 0.53,
            }
        }
    },

    "ET":{
        "ET SAS function":{
            "func": 'kumaraswamy',
            "args": {
                "a": 1.,
                "b": 1.,
                "loc": 0.0,
                "scale": 737.,
            }
        }
    }
}

sas_specs_invariant_q_g_et_e={
    
    "Q":{
        "Q SAS function":{
            "func": "gamma",
            "args": {
                "loc": 0.0,
                "scale": 8930.,
                "a": 0.53,
            }
        }
    },

    "ET":{
        "ET SAS function":{
            "func": "gamma",
            "args": {
                "loc": 0.0,
                "scale": 490.,
                "a": 1.
            }
        }
    }
}
sas_specs_storage_q_u_et_u={
    
    "Q":{
        "Q SAS function":{
            "func": 'kumaraswamy',
            "args": {
                "a": 1.,
                "b": 1.,
                "loc": 0.0,
                "scale": "S_scale",               
            }
        }
    },

    "ET":{
        "ET SAS function":{
            "func": 'kumaraswamy',
            "args": {
                "a": 1.,
                "b": 1.,
                "loc": 0.0,
                "scale": 518.,
            }
        }
    }
}
sas_specs_storage_q_g_et_u={
    
    "Q":{
        "Q SAS function":{
            "func": "gamma",
            "args": {
                "loc": 0.0,
                "scale": "S_scale", 
                "a": 0.69,

            }
        }
    },

    "ET":{
        "ET SAS function":{
            "func": 'kumaraswamy',
            "args": {
                "a": 1.,
                "b": 1.,
                "loc": 0.0,
                "scale": 398.,
            }
        }
    }
}

sas_specs_flow_q_g_et_u={
    
    "Q":{
        "Q SAS function":{
            "func": "gamma",
            "args": {
                "loc": 0.0,
                "scale": 8215.,
                "a": 0.53,

            }
        }
    },

    "ET":{
        "ET SAS function":{
            "func": 'kumaraswamy',
            "args": {
                "a": 1.,
                "b": 1.,
                "loc": 0.0,
                "scale": 737.,
            }
        }
    }
}



solute_parameters={
        "C in":{
            "C_old": 7.11,
            "alpha": {"Q": 1., "ET": 0.},
            "observations": "C out"
        }
    }

options = {
        "influx": "J",
        "verbose": True,
        "n_substeps": 1,
        "record_state": True
    }

obs_uncertainty = {
    "sigma observed C in ":{
        "prior_dis": "normal",
        "prior_params": [10e-6, 0.1],
        "is_non_negative": True,
    },
    "sigma filled C in":{
        "prior_dis": "normal",
        "prior_params": [10e-6, 5.],
        "is_non_negative": True,
    },
    "sigma C out":{
        "prior_dis": "normal",
        "prior_params": [10e-6, 0.1],
        "is_non_negative": True,
    }
}
# %%
theta_invariant_q_u_et_u = {
    "sas_specs": sas_specs_invariant_q_u_et_u,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty
}

theta_invariant_q_g_et_u = {
    "sas_specs": sas_specs_invariant_q_g_et_u,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty
}

theta_invariant_q_g_et_e = {
    "sas_specs": sas_specs_invariant_q_g_et_e,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty
}

theta_storage_q_u_et_u = {
    "sas_specs": sas_specs_storage_q_u_et_u,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty
}

theta_storage_q_g_et_u = {
    "sas_specs": sas_specs_storage_q_g_et_u,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty
}

theta_flow_q_g_et_u = {
    "sas_specs": sas_specs_flow_q_g_et_u,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty
}





# %%
