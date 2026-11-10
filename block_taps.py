"""Tap the parts of MambaLMHeadModel that sit OUTSIDE the Mamba-3 mixer: the residual
stream, the SwiGLU MLP and the final norm_f.

The tap() calls in mamba_ssm/modules/mamba3.py only see the inside of the mixer, from `u`
(the mixer input, i.e. the output of block.norm) to `out` (the output of out_proj). A loss
term that shrinks the outliers of `out` could just push them into the MLP or the residual
stream, and those taps would never notice. This file adds forward hooks that call the same
tap() function, so the new sites end up in the same .pt file, go through the same
distribution.py statistics, and appear as extra rows of the same CSV. mamba_ssm itself is
not modified.

Tap sites added for every block i (keys L{i:02d}/<name>):

    resid_in     residual stream entering block i (input of block.norm; L00 = embedding)
    resid_mid    residual stream after the mixer output is added (input of block.norm2)
    mlp_in       MLP input (output of block.norm2)
    mlp_up       first half of fc1's output, the value branch y
    mlp_gate     second half of fc1's output, the gate before SiLU
    mlp_act      y * SiLU(gate), the input of fc2
    mlp_out      output of fc2, which is added to the residual stream

and once for the whole model (bare keys, no layer):

    resid_final  residual stream after the last block (input of norm_f)
    norm_f_out   output of norm_f, the input of lm_head

Every site has d_model or d_intermediate channels on its last dimension, which is tap.py's
default channel layout, so tap.py needs no new entries.

Used by collect_distribution.py --block-taps.
"""

from mamba_ssm.utils import tap


def _input_tap(name, layer):
    """Hook that taps the first positional input of a module."""
    def hook(module, args):
        tap.tap(name, args[0], layer=layer)
    return hook


def _output_tap(name, layer):
    """Hook that taps the output of a module."""
    def hook(module, args, output):
        tap.tap(name, output, layer=layer)
    return hook


def _fc1_tap(layer):
    """fc1 writes [y, gate] side by side; GatedMLP.forward splits it with chunk(2, dim=-1)."""
    def hook(module, args, output):
        y, gate = output.chunk(2, dim=-1)
        tap.tap("mlp_up", y, layer=layer)
        tap.tap("mlp_gate", gate, layer=layer)
    return hook


def register(model):
    """Attach the hooks to a MambaLMHeadModel and return their handles."""
    if model.config.fused_add_norm:
        # With fused_add_norm the Block calls layer_norm_fn directly instead of the norm
        # modules, so the norm hooks below would never fire.
        raise RuntimeError("block_taps needs fused_add_norm=False")

    backbone = model.backbone
    handles = []
    for i, block in enumerate(backbone.layers):
        handles.append(block.norm.register_forward_pre_hook(_input_tap("resid_in", i)))
        if block.mlp is not None:   # d_intermediate=0 means no MLP and no norm2
            handles.append(block.norm2.register_forward_pre_hook(_input_tap("resid_mid", i)))
            handles.append(block.norm2.register_forward_hook(_output_tap("mlp_in", i)))
            handles.append(block.mlp.fc1.register_forward_hook(_fc1_tap(i)))
            handles.append(block.mlp.fc2.register_forward_pre_hook(_input_tap("mlp_act", i)))
            handles.append(block.mlp.fc2.register_forward_hook(_output_tap("mlp_out", i)))
    handles.append(backbone.norm_f.register_forward_pre_hook(_input_tap("resid_final", None)))
    handles.append(backbone.norm_f.register_forward_hook(_output_tap("norm_f_out", None)))
    return handles


def patch_from_pretrained():
    """Register the hooks on every model that MambaLMHeadModel.from_pretrained() returns.

    lm_eval builds the model itself inside evals/lm_harness_eval.py, so this script never
    gets hold of the model object. Wrapping from_pretrained is the one place where the
    model can be reached without editing lm_harness_eval.py.
    """
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    original = MambaLMHeadModel.from_pretrained

    def from_pretrained(*args, **kwargs):
        model = original(*args, **kwargs)
        handles = register(model)
        print(f"[block_taps] registered {len(handles)} hooks", flush=True)
        return model

    MambaLMHeadModel.from_pretrained = staticmethod(from_pretrained)
