import torch
import wandb
import os
from collections import defaultdict
import tempfile
import glob
import fnmatch


def save(filepath,
         artifacts,
         metrics=None,
         wandb_artifact=False,
         wandb_artifact_suffix='',
         wandb_run=None):
    out = defaultdict(lambda: {})
    for ty, items in artifacts.items():
        assert ty in ['model', 'optimizer', 'scalar', 'config']
        for key, value in items.items():
            if ty in ['model', 'optimizer']:
                out[ty][key] = value.state_dict()
            else:
                out[ty][key] = value
    torch.save(dict(out), filepath)
    wandb.save(filepath)
    if wandb_artifact:
        assert wandb_run is not None
        artifact_name = wandb_run.id + wandb_artifact_suffix
        if metrics:
            metadata = {'metrics': metrics}
            artifact = wandb.Artifact(artifact_name,
                                      type='trained_model',
                                      metadata=metadata)
        else:
            artifact = wandb.Artifact(artifact_name, type='trained_model')

        # Add a file to the artifact's contents
        artifact.add_file(filepath)

        # Save the artifact version to W&B and mark it as the output of this run
        wandb_run.log_artifact(artifact)


def load(artifacts,
         filepath=None,
         wandb_artifact=False,
         wandb_run=None,
         wandb_artifact_name=None,
         only_include_parameters=[],
         ignore_artifact_keys=[]):
    with tempfile.TemporaryDirectory() as tempdir:
        if wandb_artifact:
            assert wandb_artifact_name
            assert filepath is None

            artifact = wandb_run.use_artifact(wandb_artifact_name,
                                              type='trained_model')
            datadir = artifact.download(root=tempdir)
            filepath = glob.glob(os.path.join(datadir, '*.pth'))[0]
        else:
            assert filepath is not None
        out = torch.load(filepath)
        for ty, items in artifacts.items():
            assert ty in ['model', 'optimizer', 'scalar', 'config']
            if ty in ignore_artifact_keys:
                print('ignore artifact', ty)
                continue
            if ty not in ['model', 'optimizer']:
                artifacts[ty] = out[ty]
            for key, value in items.items():
                if ty == 'model':
                    load_weight(
                        value,
                        out[ty][key],
                        only_include_parameters=only_include_parameters)
                if ty == 'optimizer':
                    value.load_state_dict(out[ty][key])


def load_weight(model, pretrained_dict, only_include_parameters=[]):
    model_dict = model.state_dict()
    include_parameters = [
        k for k in pretrained_dict
        if k in model_dict and model_dict[k].shape == pretrained_dict[k].shape
    ]
    if only_include_parameters:
        newly_include_parameters = []
        for p in only_include_parameters:
            newly_include_parameters.extend(
                fnmatch.filter(include_parameters, p))
        include_parameters = newly_include_parameters

    new_pretrained_dict = {
        k: val
        for k, val in pretrained_dict.items() if k in include_parameters
    }
    diff = set(pretrained_dict.keys()) - set(new_pretrained_dict.keys())
    if diff:
        print('ignored parameters')
    for key in diff:
        print(key, pretrained_dict[key].shape)
    pretrained_dict_new_param = {
        key: val
        for key, val in model_dict.items() if key not in new_pretrained_dict
    }
    if pretrained_dict_new_param:
        print('new parameters')
    for key in pretrained_dict_new_param:
        print(key, pretrained_dict_new_param[key].shape)
    new_pretrained_dict.update(pretrained_dict_new_param)
    model.load_state_dict(new_pretrained_dict)
