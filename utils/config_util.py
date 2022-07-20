import yaml
import collections


# General config
def load_config(path, default_path=None):
    ''' Loads config file.

    Args:  
        path (str): path to config file
        default_path (bool): whether to use default path
    '''
    # Load configuration from file itself
    with open(path, 'r') as f:
        cfg_special = yaml.load(f)

    # Check if we should inherit from a config
    inherit_from = cfg_special.get('inherit_from')

    # If yes, load this config first as default
    # If no, use the default_path
    if inherit_from is not None:
        cfg = load_config(inherit_from, default_path)
    elif default_path is not None:
        with open(default_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        cfg = dict()

    # Include main configuration
    update_recursive(cfg, cfg_special)

    return cfg


def update_recursive(dict1, dict2):
    ''' Update two config dictionaries recursively.

    Args:
        dict1 (dict): first dictionary to be updated
        dict2 (dict): second dictionary which entries should be used

    '''
    for k, v in dict2.items():
        if k not in dict1:
            dict1[k] = dict()
        if isinstance(v, dict):
            update_recursive(dict1[k], v)
        else:
            dict1[k] = v


def update_dict_with_options(base, unknown_args):
    def update(d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def isfloat(x):
        try:
            y = float(x)
            return True
        except:
            return False

    def isint(x):
        return x.isdigit()

    for idx, arg in enumerate(unknown_args):
        if arg.startswith('--'):
            value = unknown_args[idx + 1]
            if value == 'false':
                value = False
            elif value == 'true':
                value = True
            elif value == 'null' or value == 'none':
                value = None
            elif value.startswith('[') and value.endswith(']'):
                temp = value[1:-1]
                temp = temp.strip(' ').split(',')
                if temp[0].startswith('"') and temp[0].endswith('"'):
                    value = [s.replace('"', '') for s in temp]
                elif isint(temp[0]):
                    value = list(map(int, temp))
                elif isfloat(temp[0]):
                    value = list(map(float, temp))
                else:
                    value = temp
            elif value.startswith('{') and value.endswith('}'):
                value = dict()
            elif isinstance(value, str) and isint(value):
                value = int(value)
            elif isinstance(value, str) and isfloat(value):
                value = float(value)

            keys = arg.split('.')
            keys[0] = keys[0].replace('--', '')

            print(value, type(value))

            new_dict = {}
            parent_dict = new_dict
            for idx, key in enumerate(keys):
                if idx == len(keys) - 1:
                    if value == 'del':
                        del parent_dict[key]
                    else:
                        parent_dict[key] = value
                else:
                    child_dict = {}
                    parent_dict[key] = child_dict
                    parent_dict = child_dict

            update(base, new_dict)
    return base
