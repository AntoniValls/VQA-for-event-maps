from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import json
import pyzed.sl as sl
from transformers import (ViltProcessor, ViltForQuestionAnswering,
                         InstructBlipProcessor, InstructBlipForConditionalGeneration,
                         LlavaNextProcessor, LlavaNextForConditionalGeneration,
                         Blip2Processor, Blip2ForConditionalGeneration)
import cv2
from PIL import Image
import torch
from inout.utils import progress_bar
import numpy as np
from typing import Dict, List, Optional, Tuple


class PromptManager:
    """Manages VQA prompts from JSON configuration file."""
    
    def __init__(self, json_path: str, preset: str = "safety_critical"):
        """
        Initialize PromptManager.
        
        Args:
            json_path: Path to JSON file with prompts
            preset: Name of preset configuration to use
        """
        with open(json_path, 'r') as f:
            self.config = json.load(f)
        
        self.base_context = self.config['metadata']['base_context']
        self.preset = preset
        self.prompts = self._load_prompts()
        
    def _load_prompts(self) -> List[Dict]:
        """Load enabled prompts based on preset."""
        preset_config = self.config['prompt_presets'].get(self.preset)
        if not preset_config:
            raise ValueError(f"Preset '{self.preset}' not found")
        
        enabled_ids = preset_config['enabled_questions']
        
        # Collect all prompts
        all_prompts = []
        for category_name, category_data in self.config['prompt_categories'].items():
            for question in category_data['questions']:
                if question.get('enabled', True):
                    all_prompts.append(question)
        
        # Filter by preset
        if enabled_ids == "all":
            return all_prompts
        else:
            return [q for q in all_prompts if q['id'] in enabled_ids]
    
    def get_prompts(self) -> List[Dict]:
        """Get list of enabled prompts."""
        return self.prompts
    
    def get_full_prompt(self, question_data: Dict) -> str:
        """Get full prompt with base context."""
        return f"{self.base_context} {question_data['text']}"
    
    def get_short_label(self, question_data: Dict) -> str:
        """Get short label for display."""
        return question_data.get('short_label', question_data['id'])


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


def draw_text_with_background(img, text, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                              font_scale=0.5, font_thickness=1,
                              text_color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Draw text with a background rectangle for better visibility."""
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    
    # Draw background rectangle
    cv2.rectangle(img, (x, y - text_height - 5), (x + text_width, y + baseline), bg_color, -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    return text_height + baseline + 5


def create_display_frame(image, frame_number, answers_dict, model_name):
    """Create a display frame with the image and answers overlay."""
    # Convert PIL Image to OpenCV format if needed
    if isinstance(image, Image.Image):
        img_display = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img_display = image.copy()
    
    # Add header info
    header_text = f"Frame: {frame_number} | Model: {model_name}"
    draw_text_with_background(img_display, header_text, (10, 30),
                             font_scale=0.7, font_thickness=2,
                             text_color=(0, 255, 0), bg_color=(0, 0, 0))
    
    # Add instruction text
    draw_text_with_background(img_display, "Press any key to continue",
                             (10, img_display.shape[0] - 20),
                             font_scale=0.5, font_thickness=1,
                             text_color=(255, 255, 0), bg_color=(0, 0, 0))
    
    # Add answers
    y_offset = 70
    for prompt_short, answer_data in answers_dict.items():
        answer = answer_data['answer']
        confidence = answer_data.get('confidence')
        
        # Format the text
        if confidence is not None:
            text = f"{prompt_short}: {answer} ({confidence:.2f})"
        else:
            text = f"{prompt_short}: {answer}"
        
        # Choose color based on answer
        if 'yes' in answer.lower() or 'safe' in answer.lower():
            text_color = (0, 255, 0)  # Green
        elif 'no' in answer.lower() or 'wait' in answer.lower() or 'stop' in answer.lower():
            text_color = (0, 0, 255)  # Red
        else:
            text_color = (255, 255, 255)  # White
        
        height = draw_text_with_background(img_display, text, (10, y_offset),
                                          font_scale=0.6, font_thickness=1,
                                          text_color=text_color, bg_color=(0, 0, 0))
        y_offset += height
        
        # Scroll if too many questions
        if y_offset > img_display.shape[0] - 60:
            break
    
    return img_display


def main(svo_input_path, output_dir, model_name, prompt_preset, frame_stride):
    """Main processing function."""
    # Device setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"Using device: {device} (dtype: {base_dtype})")
    
    # I/O setup
    os.makedirs(output_dir, exist_ok=True)
    answers_path = os.path.join(output_dir, "answers.jsonl")
    print(f"Output: {answers_path}")
    
    # Load prompts
    prompt_json_path = Path(__file__).parent / "vqa_prompts.json"
    prompt_manager = PromptManager(str(prompt_json_path), preset=prompt_preset)
    prompts = prompt_manager.get_prompts()
    print(f"\nLoaded {len(prompts)} prompts from preset: {prompt_preset}")
    
    # Initialize model
    vqa_model = VQAModel(model_name, device=device, dtype=base_dtype)
    
    # ZED Camera initialization
    zed = sl.Camera()
    input_type = sl.InputType()
    init = sl.InitParameters(input_t=input_type)
    init.set_from_svo_file(svo_input_path)
    init.svo_real_time_mode = False
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.enable_right_side_measure = False
    
    if zed.open(init) != sl.ERROR_CODE.SUCCESS:
        print("ZED initialization failed")
        return 1
    
    runtime = sl.RuntimeParameters()
    nb_frames = zed.get_svo_number_of_frames()
    
    # Create window
    cv2.namedWindow("VQA Results", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("VQA Results", 1280, 720)
    
    print(f"\nProcessing SVO: {nb_frames} frames")
    print(f"Processing every {frame_stride} frame(s)\n")
    
    try:
        with open(answers_path, "a", encoding="utf-8") as ans_f:
            while True:
                err = zed.grab(runtime)
                
                if err == sl.ERROR_CODE.SUCCESS:
                    frame = zed.get_svo_position()
                    
                    if frame % frame_stride == 0:
                        # Get image
                        left = sl.Mat()
                        zed.retrieve_image(left, sl.VIEW.LEFT)
                        arr = left.get_data()
                        
                        if arr.shape[2] == 4:
                            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
                        else:
                            img_rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
                        
                        image = Image.fromarray(img_rgb)
                        
                        # Process all prompts
                        answers_dict = {}
                        print(f"\n{'='*70}")
                        print(f"FRAME {frame} - PROCESSING {len(prompts)} QUESTIONS")
                        print(f"{'='*70}")
                        
                        for question_data in prompts:
                            question_id = question_data['id']
                            question_prompt = question_data['text']
                            full_prompt = prompt_manager.get_full_prompt(question_data) # question with context
                            short_label = prompt_manager.get_short_label(question_data)

                            # Process question
                            try:
                                answer_text, confidence = vqa_model.process_question(image, full_prompt)
                            except Exception as e:
                                print(f"Error processing question '{question_id}': {e}")
                                try:
                                    answer_text, confidence = vqa_model.process_question(image, question_prompt)
                                except Exception as e2:
                                    print(f"Error processing question '{question_id}' with basic prompt")

                            # Store for display
                            if answer_text is not None:
                                answers_dict[short_label] = {
                                    'answer': answer_text,
                                    'confidence': confidence
                                }
                                
                                # Log result
                                result_obj = {
                                    "frame": int(frame),
                                    "question_id": question_id,
                                    "question": full_prompt,
                                    "answer": answer_text,
                                    "confidence": confidence,
                                    "model": model_name
                                }
                                ans_f.write(json.dumps(result_obj) + "\n")
                                ans_f.flush()
                                
                                # Print to console
                                if confidence is not None:
                                    print(f"{short_label}:\n\tQuestion: {full_prompt}\n\tAnswer: {answer_text}\n\tConfidence: ({confidence:.2f})\n")
                                else:
                                    print(f"{short_label}:\n\tQuestion: {full_prompt}\n\tAnswer: {answer_text}\n")
                        
                        print(f"{'='*70}")
                        print("Press any key to continue...")
                        
                        # Display frame
                        display_frame = create_display_frame(image, frame, answers_dict, model_name)
                        cv2.imshow("VQA Results", display_frame)
                        cv2.waitKey(0)
                    
                    # Progress bar
                    progress_bar((frame + 1) / max(1, nb_frames) * 100, 30)
                
                elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    progress_bar(100, 30)
                    print("\n\nSVO end reached. Processing complete!")
                    break
    
    finally:
        cv2.destroyAllWindows()
        zed.close()
    
    return 0


if __name__ == "__main__":
    # ============ CONFIGURATION ============
    MODEL = "vilt"  # Options: vilt, blip2, blip2-large, llava, instructblip
    PROMPT_PRESET = "boolean_only"  # Options: safety_critical, full_assessment, environment_only, crossing_focused, boolean_only
    FRAME_STRIDE = 100  # Process every N-th frame
    SEQUENCE = 15
    # =======================================
    
    print(f"="*70)
    print(f"VQA Pedestrian Navigation System")
    print(f"="*70)
    print(f"Model: {MODEL}")
    print(f"Preset: {PROMPT_PRESET}")
    print(f"Sequence: {SEQUENCE}")
    print(f"="*70)
    
    input_svo_path = f"../data/svo/IRI_{SEQUENCE:02d}.svo2"
    output_directory = f"../data/vqa_outputs/IRI_{SEQUENCE:02d}/{MODEL}_{PROMPT_PRESET}/"
    
    exit_code = main(input_svo_path, output_directory, MODEL, PROMPT_PRESET, FRAME_STRIDE)
    sys.exit(exit_code)


# Improve general instruction to specify the egomotion. 
# Add multiple choice in the prompt configuration.
# Traffic related q
