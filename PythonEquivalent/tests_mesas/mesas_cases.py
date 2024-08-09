# %%
sas_specs_invariant_q_u_et_u = {
    "Q": {
        "Q SAS function": {
            "func": "kumaraswamy",
            "args": {
                "a": 1.0,
                "b": 1.0,
                "loc": 0.0,
                "scale": 3533.0,
            },
        }
    },
    "ET": {
        "ET SAS function": {
            "func": "kumaraswamy",
            "args": {
                "a": 1.0,
                "b": 1.0,
                "loc": 0.0,
                "scale": 2267.0,
            },
        }
    },
}


sas_specs_invariant_q_g_et_u = {
    "Q": {
        "Q SAS function": {
            "func": "gamma",
            "args": {
                "loc": 0.0,
                "scale": 8215.0,
                "a": 0.53,
            },
        }
    },
    "ET": {
        "ET SAS function": {
            "func": "kumaraswamy",
            "args": {
                "a": 1.0,
                "b": 1.0,
                "loc": 0.0,
                "scale": 737.0,
            },
        }
    },
}

sas_specs_invariant_q_g_et_e = {
    "Q": {
        "Q SAS function": {
            "func": "gamma",
            "args": {
                "loc": 0.0,
                "scale": 8930.0,
                "a": 0.53,
            },
        }
    },
    "ET": {
        "ET SAS function": {
            "func": "gamma",
            "args": {"loc": 0.0, "scale": 490.0, "a": 1.0},
        }
    },
}

sas_specs_storage_q_g_et_u = {
    "Q": {
        "Q SAS function": {
            "func": "gamma",
            "use": "scipy.stats",
            "args": {
                "loc": 0.0,
                "scale": "S_scale_old",
                "a": 0.69,
            },
            "nsegment": 50,
        }
    },
    "ET": {
        "ET SAS function": {
            "func": "kumaraswamy",
            "args": {
                "a": 1.0,
                "b": 1.0,
                "loc": 0.0,
                "scale": 398.0,
            },
        }
    },
}
sas_specs_storage_q_g_et_u_changing = {
    "Q": {
        "Q SAS function": {
            "func": "gamma",
            "args": {
                "loc": 0.0,
                "scale": "S_scale",
                "a": 0.69,
            },
            "nsegment": 50,
        }
    },
    "ET": {
        "ET SAS function": {
            "func": "kumaraswamy",
            "args": {
                "a": 1.0,
                "b": 1.0,
                "loc": 0.0,
                "scale": 398.0,
            },
        }
    },
}
solute_parameters = {
    "C in": {"C_old": 7.11, "alpha": {"Q": 1.0, "ET": 0.0}, "observations": "C out"}
}

options = {"influx": "J", "verbose": True, "n_substeps": 1, "record_state": True}

obs_uncertainty = {
    # sig_u
    "sigma observed C in": {
        "prior_dis": "normal",
        "prior_params": [0.05, 0.02],  # [0.05, 0.02],
        "is_nonnegative": True,
    },
    "sigma filled C in": {
        "prior_dis": "normal",
        "prior_params": [1.0, 0.5],  # [1., 0.5]
        "is_nonnegative": True,
    },
    "sigma C out": {
        "prior_dis": "normal",
        "prior_params": [5.0, 5.0],  # [5., 2.],
        "is_nonnegative": True,
    },
}

scale_parameters = {
    "lambda": {
        "prior_dis": "normal",
        "prior_params": [-103.0, 10.3],
        "is_nonnegative": False,
    },
    "S_c": {
        "prior_dis": "normal",
        "prior_params": [48.0, 4.8],
        "is_nonnegative": True,
    },
}
# %%
theta_invariant_q_u_et_u = {
    "sas_specs": sas_specs_invariant_q_u_et_u,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty,
}

theta_invariant_q_g_et_u = {
    "sas_specs": sas_specs_invariant_q_g_et_u,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty,
}

theta_invariant_q_g_et_e = {
    "sas_specs": sas_specs_invariant_q_g_et_e,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty,
}

theta_storage_q_g_et_u = {
    "sas_specs": sas_specs_storage_q_g_et_u,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty,
}

theta_storage_q_g_et_u_changing = {
    "sas_specs": sas_specs_storage_q_g_et_u_changing,
    "solute_parameters": solute_parameters,
    "options": options,
    "obs_uncertainty": obs_uncertainty,
    "scale_parameters": scale_parameters,
}

# %%
