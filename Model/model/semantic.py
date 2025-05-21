import os
import sys
import argparse
import torch
import base64
import random
from groq import Groq
from PIL import Image
from io import BytesIO
import open_clip

def generate_descriptions(args):

    llm = args['llm']
    KEY = args['key']

    if llm == "groq":
        client = Groq(api_key=KEY)
        model = "meta-llama/llama-4-scout-17b-16e-instruct"

    # Dictionary to hold descriptions
    class_descriptions = {}

    for split in ["train", "val", "test"]:

        base_path = args.path_to_cifarfs + split
        class_names = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

        
        if args['semantics_from'] == 'images':
            # Helper: encode image to base64
            def encode_image(image_path):
                with open(image_path, "rb") as f:
                    return base64.b64encode(f.read()).decode("utf-8")
            
            for cls in class_names:
                class_folder = os.path.join(base_path, cls)
                image_files = [f for f in os.listdir(class_folder) if f.endswith(".png")]
                selected_images = random.sample(image_files, min(args['images_per_class'], len(image_files)))
            
                image_inputs = []
                for img_name in selected_images:
                    image_path = os.path.join(class_folder, img_name)
                    encoded = encode_image(image_path)
                    image_inputs.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded}"
                        }
                    })
            
                # Build the prompt
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"These 5 images are examples from the class '{cls}'. Based on these images, give a detailed visual description that summarizes the typical appearance of this class. Briefness is required, using only one paragraph"}
                        ] + image_inputs
                    }
                ]
            
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=messages,
                    )
                    description = response.choices[0].message.content.strip()
                    class_descriptions[cls] = description
                    if args['verbose']:
                        print(f"✔️ {cls}: Description generated.")
                except Exception as e:
                    print(f"❌ Error with class {cls}: {e}")
                    class_descriptions[cls] = None

    # Optional: Save results to JSON
    import json
    with open("class_descriptions.json", "w") as f:
        json.dump(class_descriptions, f, indent=2)

    # Print results
    if args['verbose']:
        for cls, desc in class_descriptions.items():
            print(f"\n🔹 {cls.upper()}:\n{desc}\n")

    return class_descriptions

def encode_descriptions(args, class_descriptions):
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")

    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
    tokenizer = open_clip.get_tokenizer('ViT-B-32')
    model = model.to(DEVICE).eval()

    # Encode the descriptions
    class_names = list(class_descriptions.keys())
    text_descriptions = list(class_descriptions.values())
    with torch.no_grad():
        text_tokens = tokenizer(text_descriptions).to(DEVICE)
        text_features = model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    semantics = {class_names[i]: text_features[i] for i in range(len(class_names))}
    return semantics


def generate_semantics(args):
    class_descriptions = generate_descriptions(args)
    return encode_descriptions(args, class_descriptions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--llm', type=str, choices=["groq"], default="groq")
    parser.add_argument('--semantics_from', type=str, default="dafault", choices=["default", "images", "description"])
    parser.add_argument('--key', type=str)
    parser.add_argument('--images_per_class', type=int, default=5)
    parser.add_argument('--verbose', default=False, action="store_true")
    args = parser.parse_args()

    print(vars(args))

    generate_semantics(args)


