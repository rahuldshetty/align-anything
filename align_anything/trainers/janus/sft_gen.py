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
    get_optimizer_grouped_parameters,
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
            
            # Aggressive freeze: ensure vision and generation components are completely disabled
            for name, module in self.model.named_modules():
                if any(x in name for x in ['vision_model', 'aligner', 'gen_vision_model', 'gen_aligner', 'gen_head']):
                    for param in module.parameters():
                        param.requires_grad = False
            
            for name, param in self.model.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False

            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"--- Parameter Report ---")
            print(f"Total parameters: {total_params / 1e6:.2f}M")
            print(f"Trainable parameters: {trainable_params / 1e6:.2f}M")
            print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
            print(f"------------------------")
            
            if trainable_params == 0:
                print("WARNING: No trainable parameters found! Check your target_modules.")
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            self.lora_enabled = True

        if self.ds_train_cfgs is None or self.ds_train_cfgs['zero_optimization']['stage'] != 3:
            # Force everything to the target dtype and move to device
            self.model = self.model.to(dtype=dtype, device=get_current_device())
            if dtype == torch.float16:
                self.model.half()
            elif dtype == torch.bfloat16:
                self.model.bfloat16()
            
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        
        # Enable gradient checkpointing for LoRA
        if self.cfgs.train_cfgs.gradient_checkpointing:
            try:
                self.model.gradient_checkpointing_enable()
            except (ValueError, AttributeError) as e:
                print(f"Warning: Model does not support standard gradient checkpointing: {e}")
                # For Janus, we can try to enable it on the language model specifically
                if hasattr(self.model, 'language_model'):
                    try:
                        self.model.language_model.gradient_checkpointing_enable()
                        print("Gradient checkpointing enabled on the language model.")
                    except Exception:
                        print("Could not enable gradient checkpointing on the language model.")

        self.processor = VLChatProcessor.from_pretrained(
            self.cfgs.model_cfgs.model_name_or_path,
        )
        self.tokenizer = self.processor.tokenizer

    def init_deepspeed_engines(self) -> None:
        """Initialize DeepSpeed engines with memory optimizations for LoRA."""
        # Only pass trainable parameters to the optimizer to avoid DeepSpeed creating
        # master weights for the entire 1.3B model.
        trainable_params = get_optimizer_grouped_parameters(
            self.model,
            self.cfgs.train_cfgs.weight_decay,
        )
        
        # Check if we should offload to CPU
        offload_optimizer = False
        if self.ds_train_cfgs is not None:
            offload_optimizer = (
                self.ds_train_cfgs.get('zero_optimization', {})
                .get('offload_optimizer', {})
                .get('device') == 'cpu'
            )

        if offload_optimizer:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            optimizer = DeepSpeedCPUAdam(
                trainable_params,
                lr=self.cfgs.train_cfgs.learning_rate,
                betas=self.cfgs.train_cfgs.adam_betas,
            )
        else:
            from deepspeed.ops.adam import FusedAdam
            optimizer = FusedAdam(
                trainable_params,
                lr=self.cfgs.train_cfgs.learning_rate,
                betas=self.cfgs.train_cfgs.adam_betas,
            )

        # Initialize DeepSpeed
        num_update_steps_per_epoch = (
            len(self.train_dataloader) + self.cfgs.train_cfgs.gradient_accumulation_steps - 1
        ) // self.cfgs.train_cfgs.gradient_accumulation_steps
        total_training_steps = self.cfgs.train_cfgs.epochs * num_update_steps_per_epoch
        num_warmup_steps = int(self.cfgs.train_cfgs.lr_warmup_ratio * total_training_steps)
        
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=self.cfgs.train_cfgs.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_training_steps,
        )

        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(
            model=self.model,
            model_parameters=[p for p in self.model.parameters() if p.requires_grad],
            optimizer=optimizer,
            config=self.ds_train_cfgs,
            lr_scheduler=lr_scheduler,
            dist_init_required=True,
        )

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
    
    # Robustly parse unknown arguments to handle both --key value and --key=value formats
    new_unparsed_args = {}
    i = 0
    while i < len(unparsed_args):
        arg = unparsed_args[i]
        if arg.startswith('--'):
            if '=' in arg:
                k, v = arg[2:].split('=', 1)
                new_unparsed_args[k] = v
                i += 1
            else:
                k = arg[2:]
                if i + 1 < len(unparsed_args) and not unparsed_args[i+1].startswith('--'):
                    v = unparsed_args[i+1]
                    new_unparsed_args[k] = v
                    i += 2
                else:
                    new_unparsed_args[k] = True
                    i += 1
        else:
            i += 1
    unparsed_args = new_unparsed_args

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
