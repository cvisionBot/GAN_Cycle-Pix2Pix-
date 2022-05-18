from torch import optim


def get_pix_optimizer(optimizer_name, params, **kwargs):
    optim_dict = {'sgd' : optim.SGD, 'adam' : optim.Adam}
    optimizer = optim_dict.get(optimizer_name)
    return optimizer(params, **kwargs), optimizer(params, **kwargs)


def get_cycle_optimizer(optimizer_name, params, **kwargs):
    optim_dict = {'sgd' : optim.SGD, 'adam' : optim.Adam}
    optimizer = optim_dict.get(optimizer_name)
    return optimizer(params, **kwargs), optimizer(params, **kwargs), optimizer(params, **kwargs), optimizer(params, **kwargs)