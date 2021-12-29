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
        'population_size': 20,
        'function_set':('add', 'sub', 'mul', 'div','sin'),
        'generations':10
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