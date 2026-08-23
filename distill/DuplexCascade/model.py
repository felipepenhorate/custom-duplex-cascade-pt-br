import torch
import torch.nn as nn
from typing import Any, Dict, Optional
from transformers import AutoModelForCausalLM, PreTrainedTokenizer
try:
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore

    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False


class Model(nn.Module):
    """Inference-oriented wrapper around `AutoModelForCausalLM`."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model_name: str = "Qwen/Qwen2-7B-Instruct",
        trust_remote_code: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,  # e.g. "sdpa" | "flash_attention_2"
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer

        from_pretrained_kwargs: Dict[str, Any] = dict(
            trust_remote_code=trust_remote_code,
            return_dict=True,
        )
        if torch_dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            from_pretrained_kwargs["attn_implementation"] = attn_implementation

        self.llm = AutoModelForCausalLM.from_pretrained(model_name, **from_pretrained_kwargs)

        # Ensure embedding matrix matches tokenizer (special tokens may be added).
        self.llm.resize_token_embeddings(len(tokenizer))

    def _resolve_lora_target_modules(self, mode: str):
        mode = str(mode).lower().strip()
        if mode == "qv":
            return ["q_proj", "v_proj"]
        if mode == "all":
            linear_attr_names = set()
            for module_name, module in self.llm.named_modules():
                if isinstance(module, nn.Linear):
                    attr = module_name.split(".")[-1]
                    if attr:
                        linear_attr_names.add(attr)
            if not linear_attr_names:
                linear_attr_names = {
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "gate_proj",
                    "up_proj",
                    "down_proj",
                }
            return sorted(list(linear_attr_names))
        raise ValueError(f"Unknown LoRA mode: {mode}")

    def enable_lora_adapter(
        self,
        mode: str = "qv",
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.05,
        bias: str = "none",
    ) -> None:
        if not _HAS_PEFT:
            raise ImportError("`peft` is required to load LoRA adapters.")

        target_modules = self._resolve_lora_target_modules(mode=mode)
        lora_cfg = LoraConfig(
            r=int(r),
            lora_alpha=int(alpha),
            lora_dropout=float(dropout),
            bias=str(bias),
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules,
            modules_to_save=["embed_tokens", "lm_head"],
        )
        self.llm = get_peft_model(self.llm, lora_cfg)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        return_logits: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        outputs = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=False,
            **kwargs,
        )

        logits = outputs.logits
        loss = None

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            vocab_size = shift_logits.size(-1)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))

        out: Dict[str, torch.Tensor] = {"loss": loss}
        if return_logits:
            out["token_logits"] = logits
        return out

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
        **kwargs
    ):
        return self.llm.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
