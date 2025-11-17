
import torch
from typing import Optional, Tuple
from PIL import Image
from transformers import (ViltProcessor, ViltForQuestionAnswering,
                         InstructBlipProcessor, InstructBlipForConditionalGeneration,
                         LlavaNextProcessor, LlavaNextForConditionalGeneration,
                         Blip2Processor, Blip2ForConditionalGeneration)
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
        self.model.to(device)
        self.model.eval()
        print(f"Model loaded successfully on {device}")
    
    def _load_model(self) -> Tuple:
        """Load model and processor based on model name."""
        if self.model_name == "vilt":
            processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
            model = ViltForQuestionAnswering.from_pretrained(
                "dandelin/vilt-b32-finetuned-vqa",
                dtype=self.dtype if self.device == "cuda" else torch.float32
            )
            
        elif self.model_name == "blip2":
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b",
                dtype=self.dtype
            )
            
        elif self.model_name == "blip2-large":
            processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
            model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-flan-t5-xl",
                dtype=self.dtype,
                device_map="auto"
            )
            
        elif self.model_name == "llava":
            processor = LlavaNextProcessor.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf"
            )
            model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-mistral-7b-hf",
                dtype=self.dtype,
                device_map="auto"
            )
            
        elif self.model_name == "instructblip":
            processor = InstructBlipProcessor.from_pretrained(
                "Salesforce/instructblip-flan-t5-xl",
                use_fast=True
            )
            model = InstructBlipForConditionalGeneration.from_pretrained(
                "Salesforce/instructblip-flan-t5-xl",
                dtype=self.dtype,
                device_map="auto"
            )
            
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return processor, model
    
    def process_question(self, image: Image.Image, prompt: str) -> Tuple[str, Optional[float]]:
        """
        Process a single VQA question.
        
        Args:
            image: PIL Image
            prompt: Question prompt
            
        Returns:
            Tuple of (answer_text, confidence)
        """
        # Prepare inputs
        if self.model_name == "llava":
            # LLaVA needs special formatting
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            prompt_formatted = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = self.processor(images=image, text=prompt_formatted, return_tensors="pt")
        else:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        
        inputs = inputs.to(device=self.device, dtype=self.dtype)
        
        # Inference
        with torch.no_grad():
            if self.model_name == "vilt":
                outputs = self.model(**inputs)
                logits = outputs.logits
                idx = logits.argmax(-1).item()
                answer_text = self.model.config.id2label[idx]
                confidence = torch.softmax(logits, dim=-1)[0, idx].item()
                
            else:  # Generative models
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False
                )
                answer_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
                
                # Clean up the answer
                if self.model_name == "llava":
                    # LLaVA includes the prompt in output, extract answer
                    answer_text = answer_text.split("[/INST]")[-1].strip()
                else:
                    # For BLIP models, extract answer after "?"
                    answer_text = answer_text.split("?")[-1].strip(" ,.;:")
                
                confidence = None
        
        return answer_text, confidence