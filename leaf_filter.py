import io
from pathlib import Path
import torch
import open_clip
from PIL import Image

# 1. Load CLIP Model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
model = model.to(device)
model.eval()

tokenizer = open_clip.get_tokenizer("ViT-B-32")

# 2. Enhanced Multi-Class Categories (with diseased leaf prompts)
categories = {
    "close-up of a single plant leaf": [
        "a close-up photograph of one individual plant leaf",
        "a clear photo of a single leaf showing its veins and edges",
        "a close-up image of an individual leaf isolated from the plant",
        "a close-up photograph of a diseased or spotted plant leaf with fungal lesions"
    ],
    "whole plant": [
        "a photograph of an entire plant growing in a pot",
        "a photograph of a tree or bush with many leaves",
        "a wide photograph of a garden full of plants"
    ],
    "flower": [
        "a close-up photograph of a flower or blossom",
        "a photograph of a bunch of flowers"
    ],
    "fruit or vegetable": [
        "a photograph of fruit growing on a plant",
        "a close-up photograph of a fruit or vegetable"
    ],
    "animal": [
        "a photograph of an animal, bird, insect, or human",
        "a photograph of a pet or wildlife"
    ],
    "document or signature": [
        "a photograph of a handwritten signature on paper",
        "a photograph of a document, book, or printed page"
    ],
    "vehicle or ordinary object": [
        "a photograph of a car, vehicle, furniture, or household item",
        "a photograph of an everyday man-made object"
    ],
    "other scene": [
        "a landscape photograph, room interior, or outdoor scenery",
        "a random photograph that does not show an individual plant leaf"
    ]
}

# 3. Pre-encode and average the text embeddings for all categories
category_names = list(categories.keys())
category_text_features = []

with torch.no_grad():
    for cat_name in category_names:
        prompts = categories[cat_name]
        tokens = tokenizer(prompts).to(device)
        feats = model.encode_text(tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        # Average the prompt embeddings for this category
        avg_feat = feats.mean(dim=0, keepdim=True)
        avg_feat /= avg_feat.norm(dim=-1, keepdim=True)
        category_text_features.append(avg_feat)

# Shape: (num_categories, embedding_dim)
category_text_features = torch.cat(category_text_features, dim=0)


# 4. Gatekeeper Function with Intelligent Feedback
def is_valid_leaf(image_input):
    """
    Returns:
        is_leaf (bool): True if image is a single leaf, False otherwise
        detected_category (str): The category CLIP detected
        confidence (float): Confidence percentage (0-100%)
        user_message (str): Helpful error message if rejected
    """
    try:
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, (bytes, bytearray)):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            return False, "unknown", 0.0, "Invalid image format."

        image_tensor = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(image_tensor)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)

            # Cosine similarity -> Softmax probabilities
            similarity = (img_feat @ category_text_features.T)[0]
            probs = (similarity * 100.0).softmax(dim=-1)

        best_idx = torch.argmax(probs).item()
        best_cat = category_names[best_idx]
        conf = probs[best_idx].item() * 100.0

        is_leaf = (best_cat == "close-up of a single plant leaf" and conf >= 35.0)

        # Generate custom user feedback
        feedback_messages = {
            "close-up of a single plant leaf": "Valid plant leaf detected.",
            "whole plant": "You uploaded a whole plant or bush. Please take a close-up photo of a single leaf.",
            "flower": "You uploaded a flower/blossom. Please take a photo of a leaf instead.",
            "fruit or vegetable": "You uploaded a fruit/vegetable. Disease detection requires a leaf photo.",
            "animal": "An animal, insect, or person was detected. Please upload a plant leaf photo.",
            "document or signature": "A document/text was detected. Please upload a plant leaf.",
            "vehicle or ordinary object": "An everyday object was detected. Please upload an agricultural crop leaf.",
            "other scene": "The scene does not show a clear individual plant leaf. Please retake the photo."
        }

        user_msg = feedback_messages.get(best_cat, "Image rejected. Please upload a clear leaf photo.")
        
        print(f"[Gatekeeper] Detected: '{best_cat}' ({conf:.1f}%) | Valid Leaf: {is_leaf}")
        return is_leaf, best_cat, conf, user_msg

    except Exception as e:
        print(f"[Error] {e}")
        return False, "error", 0.0, "Error analyzing image."



# --- Quick Test ---
if __name__ == "__main__":

    test_img = "test.jpg"

    if Path(test_img).exists():

        # Call the function defined above
        is_leaf, cat, conf, msg = is_valid_leaf(test_img)

        print(f"\nResult: {'PASSED' if is_leaf else 'REJECTED'}")
        print(f"Category: {cat}")
        print(f"Confidence: {conf:.2f}%")
        print(f"Advice: {msg}")

    else:
        print(f"Drop an image named '{test_img}' here to test.")
