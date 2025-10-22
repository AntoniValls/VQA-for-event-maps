from transformers import (
    ViltProcessor, ViltForQuestionAnswering,
    BlipProcessor, BlipForConditionalGeneration,
    AutoTokenizer, AutoModelForSeq2SeqLM
)
from PIL import Image
import torch

# ----- CONFIG -----
IMG_PATH = "../BielGlasses/datasets/BIEL/17/image_0/000064.png"
QUESTION = "Is there something blocking the sidewalk?"

# You can switch to larger models if you have GPU:
VQA_MODEL_NAME = "dandelin/vilt-b32-finetuned-vqa"
CAPTION_MODEL = "Salesforce/blip-image-captioning-base"   # "-large" if you have the VRAM
LLM_MODEL = "google/flan-t5-base"                         # "flan-t5-large" if you have GPU

def vqa_answer(image, question):
    vqa_proc = ViltProcessor.from_pretrained(VQA_MODEL_NAME)
    vqa_model = ViltForQuestionAnswering.from_pretrained(VQA_MODEL_NAME)
    enc = vqa_proc(images=image, text=question, return_tensors="pt")
    with torch.no_grad():
        out = vqa_model(**enc)
        idx = out.logits.argmax(-1).item()
        return vqa_model.config.id2label[idx]

def caption_image(image):
    cap_proc = BlipProcessor.from_pretrained(CAPTION_MODEL)
    cap_model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL)
    inputs = cap_proc(images=image, return_tensors="pt")
    with torch.no_grad():
        ids = cap_model.generate(**inputs, max_new_tokens=40)
    return cap_proc.tokenizer.decode(ids[0], skip_special_tokens=True).strip()

def expand_answer(question, short_answer, caption):
    tok = AutoTokenizer.from_pretrained(LLM_MODEL)
    llm = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)

    prompt = (
        "You are a careful visual assistant. "
        "Given a user question about an image, a short VQA answer, "
        "and a caption describing the image, write a grounded 2–4 sentence answer. "
        "Be specific but do not invent details beyond the caption. "
        f"\n\nQuestion: {question}"
        f"\nShort answer: {short_answer}"
        f"\nCaption: {caption}"
        "\n\nLong answer:"
    )

    inputs = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        out_ids = llm.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=True, top_p=0.9, temperature=0.7
        )
    return tok.decode(out_ids[0], skip_special_tokens=True).strip()

def main():
    image = Image.open(IMG_PATH).convert("RGB")

    short = vqa_answer(image, QUESTION)
    cap = caption_image(image)
    long_ans = expand_answer(QUESTION, short, cap)

    print("Question     :", QUESTION)
    print("Short answer :", short)
    print("Caption      :", cap)
    print("\nLong answer  :", long_ans)

if __name__ == "__main__":
    main()
