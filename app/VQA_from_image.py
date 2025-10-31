from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from transformers import ViltProcessor, ViltForQuestionAnswering, Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import torch
from viz.viz_utils import save_preview

def vilt(text, image):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # be explicit with keywords
    encoding = processor(images=image, text=text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        print("Predicted answer:", model.config.id2label[idx])

    return


def blip2(text, image):
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", use_fast=True)
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", dtype=torch.float16).to(device="cuda")
    
    # Prompt template helps BLIP-2 stay in VQA mode
    prompt = f"Question: {text} Answer:"

    # be explicit with keywords
    encoding = processor(images=image, text=prompt, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    with torch.no_grad():
        output = model.generate(**encoding)
        answer = processor.decode(output[0], skip_special_tokens=True)
        print("Predicted answer:", answer)

def main(model="vilt"):
    path = "../data/images/IRI_00/image_0/000251.png"
    image = Image.open(path).convert("RGB")
    save_preview(image)                # instead of image.show()

    text = "Describe this image."

    if model == "vilt":
        vilt(text, image)
    elif model == "blip2":
        blip2(text, image)
    else:
        raise ValueError(f"Unknown model: {model}. Choose 'vilt' or 'blip2'.")

if __name__ == "__main__":
    main("blip2")
