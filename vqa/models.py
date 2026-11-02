
import torch
from typing import Optional, Tuple
from PIL import Image
import re

from transformers import (AutoProcessor, Qwen2VLForConditionalGeneration, Qwen3VLForConditionalGeneration, ViltProcessor, ViltForQuestionAnswering,
                         InstructBlipProcessor, InstructBlipForConditionalGeneration,
                         LlavaNextProcessor, LlavaNextForConditionalGeneration,
                         Blip2Processor, Blip2ForConditionalGeneration,
                         Glm4vForConditionalGeneration, Gemma4ForConditionalGeneration)

MULTIMODAL_MODELS = {
    "vilt": {
        "processor": ViltProcessor,
        "model": ViltForQuestionAnswering,
        "checkpoint": "dandelin/vilt-b32-finetuned-vqa",
        "type": "classification",
        "device_map": None,
        "attn_implementation": None

    },
    "llava": {
        "processor": LlavaNextProcessor,
        "model": LlavaNextForConditionalGeneration,
        "checkpoint": "llava-hf/llava-v1.6-mistral-7b-hf",
        "type": "chat",
        "device_map": "auto",
        "attn_implementation": None

    },
    "instructblip": {
        "processor": InstructBlipProcessor,
        "model": InstructBlipForConditionalGeneration,
        "checkpoint": "Salesforce/instructblip-flan-t5-xl",
        "type": "generation",
        "device_map": "auto",
        "attn_implementation": None

    },
    "qwen-vl": {
        "processor": AutoProcessor,
        "model": Qwen2VLForConditionalGeneration,
        "checkpoint": "Qwen/Qwen2-VL-7B-Instruct",
        "type": "chat",
        "device_map": "auto",
        "attn_implementation": "sdpa"
    },
    "qwen3-vl": {
        "processor": AutoProcessor,
        "model": Qwen3VLForConditionalGeneration,
        "checkpoint": "Qwen/Qwen3-VL-8B-Instruct",
        "type": "chat",
        "device_map": "auto",
        "attn_implementation": "sdpa",
        # Caps image tokens (~1MP worth) - this transformers release's vision tower
        # scales ~linearly with patch count and is far slower per-patch than Qwen2-VL's,
        # so uncapped (native ~3-4MP) images take 30s+ just to encode.
        "processor_kwargs": {"size": {"longest_edge": 1024 * 1024, "shortest_edge": 65536}}
    },
    "glm-4.1v": {
        "processor": AutoProcessor,
        "model": Glm4vForConditionalGeneration,
        "checkpoint": "zai-org/GLM-4.1V-9B-Thinking",
        "type": "chat",
        "device_map": "auto",
        "attn_implementation": "sdpa",
        "max_new_tokens": 1024  # needs room for the <think> trace before it reaches <answer>
    },
    "gemma-4": {
        "processor": AutoProcessor,
        "model": Gemma4ForConditionalGeneration,
        "checkpoint": "google/gemma-4-E4B-it",
        "type": "chat",
        "device_map": "auto",
        "attn_implementation": "eager"  # its audio submodule doesn't support sdpa yet
    }
}

class VQAModel:
    """Wrapper for different VQA model implementations."""
    
    def __init__(self, model_name: str, device: str = "cuda", dtype=torch.float16):
        """
        Initialize VQA model.
        
        Args:
            model_name: One of 'vilt', 'blip2', 'blip2-large', 'llava', 'instructblip'
            device: Device to run on
            dtype: Data type for model
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        print(f"Loading {model_name} model...")
        self.processor, self.model = self._load_model()
        if not hasattr(self.model, "hf_device_map"):
            self.model.to(device)
        self.model.eval()
        print(f"Model loaded successfully on {device}")
    
    def _load_model(self):
        cfg = MULTIMODAL_MODELS[self.model_name]

        processor = cfg["processor"].from_pretrained(cfg["checkpoint"], **cfg.get("processor_kwargs", {}))

        model_kwargs = {"torch_dtype": self.dtype}
        if cfg["device_map"] is not None:
            model_kwargs["device_map"] = cfg["device_map"]

        if cfg["attn_implementation"] is not None:
            model_kwargs["attn_implementation"] = cfg["attn_implementation"]

        model = cfg["model"].from_pretrained(
            cfg["checkpoint"],
            **model_kwargs
        )

        self.model_type = cfg["type"]
        return processor, model
        
    def process_question(
        self,
        image: Image.Image,
        prompt: str,
        ) -> Tuple[str, Optional[float]]:
        """
        Unified VQA inference supporting Qwen2.5-VL, LLaVA, BLIP, ViLT
        """

        # -------------------------------------------------
        # Qwen2-VL / Qwen3-VL / GLM-4.1V / Gemma 4
        # -------------------------------------------------
        if self.model_name in ("qwen-vl", "qwen3-vl", "glm-4.1v", "gemma-4"):
            cfg = MULTIMODAL_MODELS[self.model_name]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=cfg.get("max_new_tokens", 128),
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            # ✨ Mandatory trimming
            trimmed_ids = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            answer_text = self.processor.batch_decode(
                trimmed_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0].strip()

            # Thinking models (e.g. GLM-4.1V-Thinking) emit <think>...</think> before
            # the real answer, optionally wrapped in <answer>...</answer>. Keep only
            # the final answer; no-op for models that don't use these tags.
            answer_match = re.search(r"<answer>(.*?)</answer>", answer_text, re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
            else:
                answer_text = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.DOTALL).strip()

            return answer_text, None

        # -------------------------------------------------
        # LLaVA / IDEFICS / other chat models
        # -------------------------------------------------
        if self.model_type == "chat":
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            prompt_formatted = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                images=image,
                text=prompt_formatted,
                return_tensors="pt",
            )

        # -------------------------------------------------
        # BLIP / InstructBLIP / ViLT
        # -------------------------------------------------
        else:
            inputs = self.processor(
                images=image,
                text=prompt,
                return_tensors="pt",
            )

        # -------------------------------------------------
        # Device placement
        # -------------------------------------------------
        if hasattr(self.model, "hf_device_map"):
            inputs = {
                k: v.to(self.model.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
        else:
            inputs = inputs.to(device=self.device, dtype=self.dtype)

        # -------------------------------------------------
        # Inference
        # -------------------------------------------------
        with torch.no_grad():

            if self.model_type == "classification":
                outputs = self.model(**inputs)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                answer_text = self.model.config.id2label[idx]
                confidence = torch.softmax(logits, dim=-1)[0, idx].item()
                return answer_text, confidence

            gen_kwargs = dict(
                max_new_tokens=128,
                do_sample=False,
            )

            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer and tokenizer.pad_token_id is None:
                gen_kwargs["pad_token_id"] = tokenizer.eos_token_id

            output_ids = self.model.generate(**inputs, **gen_kwargs)

            decoded = self.processor.decode(
                output_ids[0],
                skip_special_tokens=True,
            )

        answer_text = decoded.strip()

        for sep in ["ASSISTANT:", "Assistant:", "[/INST]"]:
            if sep in answer_text:
                answer_text = answer_text.split(sep)[-1].strip()

        return answer_text, None
