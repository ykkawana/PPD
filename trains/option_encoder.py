import yaml


def encode(opts, opt_combo_config_path):
    opt_combo_config = yaml.safe_load(open(opt_combo_config_path))['combo']
    new_opts = opts.copy()
    for idx, opt in enumerate(opts):
        if opt.startswith('--combo.'):
            opt_combo_name = opt.split('.')[1]
            for arg in opt_combo_config[opt_combo_name][opts[idx + 1]]:
                new_opts.extend(arg.split(' '))
    return new_opts
