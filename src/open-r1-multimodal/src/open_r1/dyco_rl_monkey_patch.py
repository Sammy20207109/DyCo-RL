"""
DyCo-RL monkey patches: extended output dataclasses + local generation_utils.
"""
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, List

from transformers.utils import ModelOutput


@dataclass
class BaseModelOutputWithPastDyCo(ModelOutput):
    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    img_ratio: Optional[torch.FloatTensor] = None
    text_ratio: Optional[torch.FloatTensor] = None
    img_attn: Optional[torch.FloatTensor] = None
    text_attn: Optional[torch.FloatTensor] = None


@dataclass
class Qwen2_5_VLCausalLMOutputWithPastDyCo(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None
    img_ratio: Optional[torch.FloatTensor] = None
    text_ratio: Optional[torch.FloatTensor] = None
    img_attn: Optional[torch.FloatTensor] = None
    text_attn: Optional[torch.FloatTensor] = None


def apply_dyco_rl_patches():
    """Replace library output classes + GenerationMixin with local versions."""
    import transformers.modeling_outputs as mo
    mo.BaseModelOutputWithPast = BaseModelOutputWithPastDyCo

    import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_mod
    qwen_mod.Qwen2_5_VLCausalLMOutputWithPast = Qwen2_5_VLCausalLMOutputWithPastDyCo

    # Replace generate/_sample with local generation_utils.py
    from open_r1.generation_utils import GenerationMixin as LocalGenMixin
    import transformers.generation.utils as gen_mod

    gen_mod.GenerationMixin.generate = LocalGenMixin.generate
    gen_mod.GenerationMixin._sample = LocalGenMixin._sample

    # Preserve staticmethod descriptor for _batched_M_FR
    raw_mfr = LocalGenMixin.__dict__.get('_batched_M_FR')
    if raw_mfr is not None:
        gen_mod.GenerationMixin._batched_M_FR = raw_mfr

    # Patch concrete model class (already resolved MRO)
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
    Qwen2_5_VLForConditionalGeneration.generate = LocalGenMixin.generate
    Qwen2_5_VLForConditionalGeneration._sample = LocalGenMixin._sample
    if raw_mfr is not None:
        Qwen2_5_VLForConditionalGeneration._batched_M_FR = raw_mfr

    print("[DyCo-RL] Patches applied: output classes + generation_utils")
