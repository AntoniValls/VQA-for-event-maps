from transformers import ViltProcessor, ViltForQuestionAnswering
from PIL import Image
import torch
from utils.utils import save_preview

def main():
    path = "../BielGlasses/datasets/BIEL/17/image_0/000064.png"
    image = Image.open(path).convert("RGB")
    save_preview(image)                # instead of image.show()

    text = "Is there something blocking the sidewalk?"

    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # be explicit with keywords
    encoding = processor(images=image, text=text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        print("Predicted answer:", model.config.id2label[idx])

if __name__ == "__main__":
    main()
