def get_optim_params():
    opt_params = {'lr':0.5}
    dic = {
        'alpha':0.1,
        'beta':0.1,
        'optim_name':'sgd',
        'optim_params':opt_params,
        'num_epochs':200,
        'patience':20}

    return dic

def get_gp_params():
    dic = {
        'population_size': 5000,
        'function_set':('add', 'sub', 'mul', 'div', 'sin', 'exp','log'),
        'generations':20,
        'parsimony_coefficient': 1e-20,
        'tournament_size':50,
        # 'p_crossover': 0.206322262,
        # 'p_subtree_mutation':0.053164874,
        # 'p_hoist_mutation':0.368744818,
        # 'p_point_mutation':0.257603826
    }
    return dic

def get_tvdiff_params():
    return {
        'alph':0.01,
        'itern': 50,
        'plotflag': False,
        'precondflag': False,
        'scale':'small'
    }

def get_trenddiff_params():
    return {
        'order':0,
        'alpha':0.01
    }

def get_splinediff_params():
    return {
        's': 0.01
    }

def get_finitediff_params():
    return {
        'k': 1
    }