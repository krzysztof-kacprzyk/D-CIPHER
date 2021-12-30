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
        'population_size': 1000,
        'function_set':('add', 'sub', 'mul', 'div','sin'),
        'generations':20,
        'parsimony_coefficient':1.0
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