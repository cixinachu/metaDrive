"""Shared constructor options; each algorithm module owns its model factory."""
import torch


def sac_options(c):
    return dict(learning_rate=c['learning_rate'], batch_size=c['batch_size'], buffer_size=c['buffer_size'],
                learning_starts=c['learning_starts'], gamma=c['gamma'], tau=c['tau'], seed=c['seed'], device=c['device'],
                policy_kwargs=dict(net_arch=[c['hidden_size'], c['hidden_size']], activation_fn=torch.nn.LeakyReLU))


def constrained_options(c):
    options=sac_options(c)
    options.update({k:c[k] for k in ('cost_limit','gamma_cost','tail_fraction','safety_lr','dual_lr','initial_multiplier','n_quantiles')})
    options['safety_width']=c['hidden_size']
    return options
