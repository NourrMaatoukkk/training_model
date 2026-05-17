import os
import io
import json
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from PIL import Image
import streamlit as st


# ── Inference transform must match val_tf in train.py ──────────────────────────
INFER_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


@st.cache_resource
def load_classes(dataset_path='dataset'):
    """Load class names: JSON first (works locally), fall back to dataset folder."""
    if os.path.exists('class_names.json'):
        with open('class_names.json') as f:
            return json.load(f)
    if os.path.exists(dataset_path):
        try:
            return datasets.ImageFolder(root=dataset_path).classes
        except Exception:
            pass
    return None


@st.cache_resource
def load_model(model_path, num_classes):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    state = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state)
    model.eval()
    return model


def predict(model, img_pil, classes, topk=3):
    tensor = INFER_TRANSFORM(img_pil).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    k = min(topk, len(classes))
    vals, idxs = torch.topk(probs, k)
    return [(classes[i], float(v)) for i, v in zip(idxs.tolist(), vals.tolist())]


def main():
    st.set_page_config(page_title='Sea Life Classifier', layout='centered')
    st.title('🐠 Sea Life Classifier')
    st.caption('ResNet18 — transfer learning')

    # ── Sidebar: file paths ────────────────────────────────────────────────────
    with st.sidebar:
        st.header('Model files')
        model_path  = st.text_input('Checkpoint path', value='best_model.pth')
        classes_path = st.text_input('class_names.json path', value='class_names.json')

    # ── Load class names ───────────────────────────────────────────────────────
    classes = None
    if os.path.exists(classes_path):
        with open(classes_path) as f:
            classes = json.load(f)
    else:
        classes = load_classes()

    if classes is None:
        st.error('class_names.json not found. Make sure it is in the same folder.')
        st.stop()

    # ── Load model ─────────────────────────────────────────────────────────────
    if not os.path.exists(model_path):
        st.warning(f'`{model_path}` not found. Upload it below.')
        uploaded_pth = st.file_uploader('Upload best_model.pth', type=['pth'])
        if uploaded_pth:
            with open(model_path, 'wb') as f:
                f.write(uploaded_pth.getbuffer())
            st.success('Saved. Refresh the page.')
        st.stop()

    try:
        model = load_model(model_path, len(classes))
    except Exception as e:
        st.error(f'Failed to load model: {e}')
        st.stop()

    st.success(f'Model loaded — {len(classes)} classes')

    # ── Image upload & prediction ──────────────────────────────────────────────
    uploaded_img = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])
    if uploaded_img is None:
        st.info('Upload an image to get a prediction.')
        return

    image = Image.open(io.BytesIO(uploaded_img.read())).convert('RGB')
    st.image(image, caption='Input image', use_column_width=True)

    with st.spinner('Running inference...'):
        results = predict(model, image, classes, topk=3)

    st.subheader('Top predictions')
    for label, prob in results:
        st.write(f'**{label}** — {prob*100:.1f}%')
        st.progress(prob)

    st.bar_chart({label: prob for label, prob in results})


if __name__ == '__main__':
    main()