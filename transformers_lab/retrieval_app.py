import gradio as gr
import torch
from transformers.models.clip.modeling_clip import CLIPModel
from transformers.models.clip.processing_clip import CLIPProcessor
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn.functional as F

model_id = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_id, dtype=torch.bfloat16)
processor = CLIPProcessor.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = load_dataset("jxie/flickr8k", split="train")

def collate_fn(batch):
    images = [item["image"] for item in batch]
    
    input = processor(images=images, return_tensors="pt")
    return input

dataloader = DataLoader(dataset=dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=4)

img_embeddings = []
with torch.no_grad():
    for inputs in dataloader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        img_features = model.get_image_features(**inputs)

        img_features = F.normalize(img_features, p=2, dim=-1)
        img_embeddings.append(img_features.cpu())

img_embeddings = torch.vstack(img_embeddings)


def retrieve_images(text_prompt, top_k):
    with torch.no_grad():
 
        inputs = processor(text=text_prompt, return_tensors="pt").to(device)
        text_features = model.get_text_features(**inputs)
        text_features = F.normalize(text_features, p=2, dim=-1).cpu()
        
        similarities = torch.matmul(text_features, img_embeddings.T).squeeze(0)
        
        top_indices = similarities.topk(top_k).indices.tolist()
        
        results = dataset.select(top_indices)["image"]
        
    return results

# GUI
interface = gr.Interface(
    fn=retrieve_images,
    inputs=[gr.Textbox(lines=1, placeholder="Es. a photo of dogs playing in the snow...", label="Query"), gr.Slider(minimum=1, maximum=25, step=1)],
    outputs=gr.Gallery(columns=5, height="auto"),
    title="A Text-to-image Retrieval System using CLIP on Flickr8k",
    description="Insert a photo description.",
    live=True,
    flagging_mode="never"
)

if __name__ == "__main__":
    interface.launch()