from typing import List, Mapping, Optional
import torch
import re


def load_pretrained(
    model: torch.nn.Module,
    pretrained_params: Mapping[str, torch.Tensor],
    strict: bool = False,
    prefix: Optional[List[str]] = None,
    verbose: bool = True,
) -> torch.nn.Module:
    """
    Load pretrained weights to the model.

    Args:
        model (torch.nn.Module): The model to load weights to.
        pretrained_params (Mapping[str, torch.Tensor]): The pretrained parameters.
        strict (bool): Whether to strictly enforce that the keys in :attr:`pretrained_params`
            match the keys returned by this module's :meth:`Module.state_dict`. Default to False.
        prefix (List[str]): The list of prefix strings to match with the keys in
            :attr:`pretrained_params`. The matched keys will be loaded. Default to None.

    Returns:
        torch.nn.Module: The model with pretrained weights.
    """

    model_new_params = model.state_dict()

    not_identical = []
    updated = []
    for key in pretrained_params.keys():
        if key in model_new_params.keys() and model_new_params[key].shape == pretrained_params[key].shape:
            model_new_params[key] = pretrained_params[key]
            updated.append(key)
        else:
            if key in model_new_params.keys():
                if verbose:
                    print(
                        f"Same key but not same shape: {model_new_params[key].shape} : {pretrained_params[key].shape}")
            not_identical.append(key)

    def modify_string(input_string):
        # Pattern to check if the string has the required format
        check_pattern = r"^encoder\.layers\.\d+\.self_attn\.self_attn\..*"

        # Check if the input string matches the pattern
        if re.match(check_pattern, input_string):
            # Pattern to perform the replacement
            modify_pattern = r"(encoder\.layers\.\d+\.self_attn\.)self_attn\.(.*)"

            # Replace the matched pattern with the required modification
            modified_string = re.sub(modify_pattern, r"\1\2", input_string)
            return modified_string
        else:
            return input_string

    for key in not_identical[:]:
        key_new = modify_string(key)
        if key_new in model_new_params.keys() and model_new_params[key_new].shape == pretrained_params[key].shape:
            model_new_params[key_new] = pretrained_params[key]
            updated.append(key_new)
            not_identical.remove(key)

    if verbose and len(not_identical) > 0:
        print("Couldn't load following keys: ", not_identical)

    model.load_state_dict(model_new_params)
    return model
