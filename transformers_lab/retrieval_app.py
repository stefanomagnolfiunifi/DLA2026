import gradio as gr
import torch
from transformers import AutoProcessor, AutoModel
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn.functional as F

model_id = "openai/clip-vit-base-patch32"
model = AutoModel.from_pretrained(model_id, dtype=torch.bfloat16)
processor = AutoProcessor.from_pretrained(model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = load_dataset("jxie/flickr8k", split="train")
images = dataset['image']
#dataloader = DataLoader(dataset=images, batch_size=64, shuffle=False)

img_embeddings = []
with torch.no_grad():
    for img in images:
        inputs = processor(images=img, return_tensors="pt").to(device)
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
        
        results = [images[idx] for idx in top_indices]
        
    return results

# GUI
interface = gr.Interface(
    fn=retrieve_images,
    inputs=[gr.Textbox(lines=1, placeholder="Es. a photo of dogs playing in the snow...", label="Query"), gr.Slider(minimum=1, maximum=25, step=1)],
    outputs=gr.Gallery(columns=5, height="auto"),
    title="A Text-to-image Retrieval System using CLIP on Flickr8k",
    description="Insert a photo description."
)

if __name__ == "__main__":
    interface.launch()