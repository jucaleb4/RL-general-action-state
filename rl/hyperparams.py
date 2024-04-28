import warnings
from collections import namedtuple

INF = float('inf')

available_envs = [
    "gym_examples/GridWorld-v0", 
    "LunarLander-v2", 
    "MountainCar-v0", 
    "Pendulum-v1",
    "Humanoid-v4",
    "HalfCheetah-v4",
    "VMPacking-v0",
]

available_algs = [
    "dqn",
    "pmd", 
    "ppo",
]

required_keys = [
    'save_fname',       
    'seed',       
    'env',              
    'gamma',            
    'alg',              
    'max_iter',         
    'max_ep',           
    'max_step',         
    'rollout_len',      
    'norm_obs',         
    'norm_rwd',         
    'use_adv',          
    'norm_sa_val',      
    'max_grad_norm',    
    'po_stepsize',      
    'po_base_stepsize', 
    'pe_stepsize',      
    'pe_base_stepsize', 
    'pe_max_iter',      
    "minibatch_size",   
    'fa_type',          
    'pe_reg',           
    "pe_method",        
    'parallel',         
    'max_trial',        
]

def isnumber(x):
    return isinstance(x, int) or isinstance(x, float)

Setting = namedtuple('Setting', [
    'save_fname',       # Logging file. Must be empty or a valid string
    'seed',       
    'env',              # Environment name to pass into Gymnasium
    'gamma',            # discount factor
    'alg',              # algorithm to run
    'max_iter',         # max number of training iters
    'max_ep',           # max number of episodes
    'max_step',         # max number of steps 
    'rollout_len',      # number of samples for policy evaluation
    'norm_obs',         # normalize observation to 0 mean and unit variance
    'norm_rwd',         # normalize reward to unit variance
    'use_adv',          # use advantage for state-action value
    'norm_sa_val',      # normalize state-action value 
    'max_grad_norm',    # l_∞ norm clip value for neural network gradient
    'po_stepsize',      # policy opt stepsize (constant or decr)
    'po_base_stepsize', # policy opt base stepsize
    'pe_stepsize',      # policy eval stepsize 
    'pe_base_stepsize', # policy eval base stepsize. Accepts -1 for default val
    'pe_max_iter',      # max number of eval iterations
    "minibatch_size",   # applies minibatch, if applicable
    'fa_type',          # function approximation type
    'pe_reg',           # policy eval regularization (e.g., weight decay). Accepts -1 for default value
    "pe_method",        # policy eval method (adam, sgd)
    'parallel',         # run experiments in parallel (via multi-threading)
    'max_trial',        # number of trials/experiments to run
])

def create_and_validate_settings(d: dict) -> tuple[Setting, bool]:
    """ 
    Parses through a dictionary to check hyperparameters for valid values.
    First we check through required keys mandated by `required_keys`. Then
    we check the values are valid for the algorithm.

    :return setting: namedtuple of settings
    :return succeed: whether the parameters were valid
    """ 
    for key in required_keys:
        if key not in d:
            print("Missing key %s" % key)
            return None, False

    if d['save_fname'] is not None and isinstance(d['save_fname'], str) and len(d['save_fname']) > 0:
        ext = ".csv"
        if len(d['save_fname']) < 4 or d['save_fname'][-len(ext):] != ext:
            print("Received invalid save filename %s" % d['save_fname'])
            return None, False
    
    if not (d['seed'] is None or isinstance(seed, int)):
        print("Received invalid seed %s; must be None or int" % d['seed'])
        return None, False
            
    if d['env'] not in available_envs:
        print("Unavailable env %s" % d['env'])
        return None, False
        
    if not(isnumber(d['gamma']) and 0 <= d['gamma'] <= 1):
        print("Invalid discount factor (gamma) %s" % d['gamma'])
        return None, False

    if d['alg'] not in available_algs:
        if d['alg'] == "pda":
            print("alg PDA is not available; run alg PMD instead")
            return None, False
        else:
            print("Invalid algorithm %s" % d['alg'])
            return None, False

    max_len_keys = ["iter", "ep", "step"]
    for key in max_len_keys:
        full_key = "max_%s" % key
        val = d.get(full_key)
        if not(isinstance(val, int) and (val == -1 or val > 0)):
            print("Invalid max_%s %s, must be -1 or a positive integer" % (full_key, val))
            return None, False
        if val == -1:
            d[full_key] = INF

    if d['max_iter'] == d['max_ep'] == d['max_step'] == INF:
        print("Cannot have max_iter, max_ep, and max_step all be -1 (infinite)")
        return None, False

    if d['alg'] == "pmd":
        if not(isinstance(d['rollout_len'], int) and d['rollout_len'] > 0):
            print("Rollout len %s must be a positive integer" % d['rollout_len'])
            return None, False

        rescale_keys = ["norm_obs", "norm_rwd", "use_adv", "norm_sa_val"]
        for key in rescale_keys:
            val = d.get(key)
            if not(isinstance(val, bool)):
                print("Invalid %s = %s, must be boolean" % (key, val))
                return None, False

        if not(isnumber(d['max_grad_norm']) and (d['max_grad_norm'] == -1 or d['max_grad_norm'] > 0)):
            print("Invalid max_grad_norm %s, must be -1 or a positive number" % (key, d['max_grad_norm']))
            return None, False
        if d['max_grad_norm'] == -1:
            d['max_grad_norm'] = INF

        opt_probs = ["po", "pe"]
        for key in opt_probs:
            full_key = "%s_base_stepsize" % key
            val = d.get(full_key)
            if not(isnumber(val) and val > 0):
                print("Invalid %s_base_stepsize %s, must be positive" % (full_key, val))
                return None, False
            
        if d['po_stepsize'] not in ["decreasing", "constant"]:
            print("Invalid po_stepsize %s, must be 'decreasing' or 'constant'" % d['po_stepsize'])
            return None, False

        if d['pe_stepsize'] not in ["constant", "optimal"]:
            print("Invalid pe_stepsize %s, must be 'constant' or 'optimal'" % d['pe_stepsize'])
            return None, False

        if not(isnumber(d['pe_reg']) and (d['pe_reg'] == -1 or d['pe_reg'] >= 0)):
            print("Invalid pe_reg %s, must be -1 or a nonnegative value" % d['pe_reg'])
            return None, False

        if not(isinstance(d['pe_max_iter'], int) and d['pe_max_iter'] > 0):
            print("Invalid pe_max_iter %s, must be a positive integer" % d['pe_max_iter'])
            return None, False

        if not(isinstance(d['fa_type'], str) and d['fa_type'] in ["linear", "nn"]):
            print("Invalid fa_type %s, must be 'linear' or 'nn' neural network)" % d['fa_type'])
            return None, False

    if not(isinstance(d['minibatch_size'], int) and d['minibatch_size'] > 0):
        print("Invalid minibatch_size %s, must be a positive integer" % d['minibatch_size'])
        return None, False

    if not(isinstance(d['pe_method'], str) and d['pe_method'] in ["sgd", "adam"]): 
        print("Invalid policy evaluation pe_method %s, must be 'sgd' or 'adam'" % d['pe_method'])
        return None, False

    if not(isinstance(d['parallel'], bool)):
        print("Invalid parallel = %s, must be boolean" % (d['parallell']))
        return None, False
            
    if not(isinstance(d['max_trial'], int)):
        print("Invalid max_trial = %s, must be boolean" % (d['max_trial']))
        return None, False

    # logical checks
    if d['alg'] == 'pmd' and d['fa_type'] == 'nn' and d['po_stepsize'] == 'constant':
        print("PMD with neural network does not support po_stepsize='constant'; please use 'decreasing'")
        return None, False

    return Setting(**d), True

raw_setting = dict({
    'save_fname': "save.csv",
    'seed': None,
    'env': "LunarLander-v2",
    'gamma': 0.99,
    'alg': "pmd",
    'max_iter': -1,
    'max_ep': -1,
    'max_step': int(1e6),
    'rollout_len': 10,
    'norm_obs': True,
    'norm_rwd': False,
    'use_adv': False,
    'norm_sa_val': False,
    'max_grad_norm': -1,
    'po_stepsize': 'decreasing',
    'po_base_stepsize': 1,
    'pe_stepsize': 'constant',
    'pe_base_stepsize': 1,
    'pe_max_iter': 1000,
    'minibatch_size': 32,
    'fa_type': 'nn',
    'pe_reg': 0,
    'pe_method': 'sgd',
    'parallel': False,
    'max_trial': 1,
})

# setting, success = create_and_validate_settings(raw_setting)
# print("Succeeded: %s" % success)
