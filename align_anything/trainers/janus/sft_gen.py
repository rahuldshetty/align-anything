# Copyright 2025 PKU-Alignment Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Trainer for supervised training."""


import argparse
import os
import sys

import deepspeed
import torch
import transformers
from janus.models import MultiModalityCausalLM, VLChatProcessor, VLMImageProcessor

from align_anything.datasets.janus import SupervisedBatch, SupervisedDataset, SupervisedTokenizedDataset
from align_anything.trainers.text_to_text.sft import SupervisedTrainer as SupervisedtextTrainer
from align_anything.utils.device_utils import torch_set_device
from align_anything.utils.multi_process import get_current_device
from align_anything.utils.tools import (
    custom_cfgs_to_dict,
    dict_to_namedtuple,
    read_cfgs,
    seed_everything,
    update_dict,
)


transformers.logging.set_verbosity_info()


class SuperviseTrainer(SupervisedtextTrainer):

    def init_datasets(self) -> None:
        """Initialize training and evaluation datasets."""
        self.train_dataloader, self.eval_dataloader = self.get_dataloaders(
            SupervisedDataset, SupervisedDataset
        )


    def update_configs(self, model_config, args, fields):
        cross_update = lambda a, b, field_name: (
            setattr(b, field_name, getattr(a, field_name))
            if getattr(b, field_name, None) is None
            else setattr(a, field_name, getattr(b, field_name))
        )

        for f in fields:
            cross_update(model_config, args, f)

    def init_models(self) -> None:
        """Initialize model and tokenizer."""
        if self.cfgs.train_cfgs.bf16:
            dtype = torch.bfloat16
        elif self.cfgs.train_cfgs.fp16:
            dtype = torch.float16
        else:
            dtype = torch.float32

        # DeepSpeed ZeRO-3 patches model __init__ to redirect all tensor
        # creation to the meta device. siglip_vit.VisionTransformer calls
        # `tensor.item()` during __init__, which raises:
        #   RuntimeError: Tensor.item() cannot be called on meta tensors
        # Fix: temporarily force torch.linspace to always produce CPU tensors
        # so .item() succeeds, then restore the original after loading.
        # Patch 1 — siglip_vit meta-tensor fix (see above).
        _original_linspace = torch.linspace

        def _cpu_linspace(*args, **kwargs):
            kwargs['device'] = 'cpu'
            return _original_linspace(*args, **kwargs)

        # Patch 2 — MultiModalityCausalLM.__init__ calls super().__init__(config)
        # without forwarding attn_implementation, so HuggingFace auto-detects and
        # tries sdpa/flash_attn — both unsupported by this model.  Monkey-patch
        # _check_and_adjust_attn_implementation to silently fall back to "eager".
        import transformers.modeling_utils as _mu
        _original_check_attn = _mu.PreTrainedModel._check_and_adjust_attn_implementation

        def _eager_fallback_check(self_inner, *args, **kwargs):
            try:
                return _original_check_attn(self_inner, *args, **kwargs)
            except (ImportError, ValueError):
                return 'eager'

        # Patch 3 — transformers 5.x added `all_tied_weights_keys` (a dict) to
        # PreTrainedModel, but Janus's MultiModalityCausalLM only has the old
        # `_tied_weights_keys` (a list).  Backfill the property so that
        # mark_tied_weights_as_initialized() can call .keys() on it.
        if not hasattr(MultiModalityCausalLM, 'all_tied_weights_keys'):
            MultiModalityCausalLM.all_tied_weights_keys = property(
                lambda self: {k: None for k in (getattr(self, '_tied_weights_keys', None) or [])}
            )

        from transformers.integrations.deepspeed import HfDeepSpeedConfig
        if self.ds_train_cfgs is not None and self.ds_train_cfgs['zero_optimization']['stage'] == 3:
            self.dstchf = HfDeepSpeedConfig(self.ds_train_cfgs)

        torch.linspace = _cpu_linspace
        _mu.PreTrainedModel._check_and_adjust_attn_implementation = _eager_fallback_check
        try:
            self.model = MultiModalityCausalLM.from_pretrained(
                self.cfgs.model_cfgs.model_name_or_path,
                torch_dtype=dtype,
                trust_remote_code=self.cfgs.model_cfgs.trust_remote_code,
                low_cpu_mem_usage=True,
            )
        finally:
            torch.linspace = _original_linspace
            _mu.PreTrainedModel._check_and_adjust_attn_implementation = _original_check_attn

        if self.cfgs.lora_cfgs and self.cfgs.lora_cfgs.use_lora:
            from peft import get_peft_model, LoraConfig
            lora_config = LoraConfig(
                r=self.cfgs.lora_cfgs.r,
                lora_alpha=self.cfgs.lora_cfgs.lora_alpha,
                target_modules=self.cfgs.lora_cfgs.target_modules,
                lora_dropout=self.cfgs.lora_cfgs.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
            self.lora_enabled = True

        if self.ds_train_cfgs is None or self.ds_train_cfgs['zero_optimization']['stage'] != 3:
            self.model = self.model.to(get_current_device())

        self.processor = VLChatProcessor.from_pretrained(
            self.cfgs.model_cfgs.model_name_or_path,
        )
        self.tokenizer = self.processor.tokenizer

    def loss(self, sft_batch: SupervisedBatch) -> dict[str, torch.Tensor]:
        """Loss function for supervised finetuning."""
        print("sft_batch", sft_batch)
        outputs = self.model.forward(**sft_batch)
        return {
            'loss': outputs.loss,
        }


def main():
    # setup distribution training
    deepspeed.init_distributed()
    current_device = get_current_device()
    torch_set_device(current_device)

    # read default configs from the yaml file
    task = os.path.join('janus', 'sft_gen')
    dict_cfgs, ds_cfgs = read_cfgs(mode='train', task=task)

    # get custom configs from command line
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _, unparsed_args = parser.parse_known_args()
    keys = [k[2:] for k in unparsed_args[0::2]]
    values = list(unparsed_args[1::2])
    unparsed_args = dict(zip(keys, values))
    for k, v in unparsed_args.items():
        dict_cfgs = update_dict(dict_cfgs, custom_cfgs_to_dict(k, v))

    # setup training
    cfgs = dict_to_namedtuple(dict_cfgs)
    seed_everything(cfgs.train_cfgs.seed)

    # finetune the model
    trainer = SuperviseTrainer(cfgs=cfgs, ds_cfgs=ds_cfgs)
    trainer.train()
    trainer.save()


if __name__ == '__main__':
    sys.exit(main())
