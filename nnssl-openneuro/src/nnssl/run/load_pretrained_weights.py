import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP


def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were obtained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        '.seg_layers.',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key].shape, \
                f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
                f"does not seem to be compatible with your network."

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)

import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP


def load_my_pretrained_weights(network, fname, verbose=False):
    """
    Loads only matching weights from a pretrained model into the current network (e.g., encoder-only MAE weights).
    Skips keys that are missing or whose shapes do not match.
    """
    saved_model = torch.load(fname, map_location='cpu')
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        '.seg_layers.',
        '.decoder',
        '.projector',
        '.predictor',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    matched_keys = []
    unmatched_keys = []
    missing_keys = []

    for key in model_dict:
        if any(s in key for s in skip_strings_in_pretrained):
            continue
        if key in pretrained_dict and model_dict[key].shape == pretrained_dict[key].shape:
            matched_keys.append(key)
        elif key in pretrained_dict:
            unmatched_keys.append((key, model_dict[key].shape, pretrained_dict[key].shape))
        else:
            missing_keys.append(key)

    if verbose:
        print("==== Matched Keys ====")
        for k in matched_keys:
            print(f"✓ {k}")
        print("\n==== Unmatched Keys ====")
        for k, shape1, shape2 in unmatched_keys:
            print(f"✗ {k}: model shape {shape1}, pretrained shape {shape2}")
        print("\n==== Missing Keys in Pretrained ====")
        for k in missing_keys:
            print(f"⚠️ {k}")

    pretrained_filtered = {k: pretrained_dict[k] for k in matched_keys}
    model_dict.update(pretrained_filtered)
    mod.load_state_dict(model_dict)

    print("✅ Successfully loaded encoder weights from:", fname)



