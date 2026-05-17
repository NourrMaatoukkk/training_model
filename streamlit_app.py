import os
import io
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
import numpy as np

import streamlit as st


@st.cache_resource
def load_classes(dataset_path='dataset'):
    if os.path.exists(dataset_path):
        try:
            ds = datasets.ImageFolder(root=dataset_path)
            return ds.classes
        except Exception:
            return None
    return None


@st.cache_resource
def load_model(model_path='resnet18_model.pth'):
    device = torch.device('cpu')
    # try to inspect saved state to infer num_classes
    if not os.path.exists(model_path):
        return None, None
    state = torch.load(model_path, map_location='cpu')
    # find the final fc weight to discover num_classes (prefer highest fc.<idx>.weight)
    import re
    candidates = []
    for k, v in state.items():
        m = re.match(r'^fc(?:\.(\d+))?\.weight$', k)
        if m and hasattr(v, 'ndim') and v.ndim == 2:
            idx = int(m.group(1)) if m.group(1) is not None else 0
            candidates.append((idx, k, v))
    num_classes = None
    if candidates:
        candidates.sort(key=lambda x: x[0])
        num_classes = candidates[-1][2].shape[0]
    # if we couldn't infer, try dataset classes
    classes = load_classes()
    if num_classes is None and classes is not None:
        num_classes = len(classes)
    if num_classes is None:
        raise RuntimeError('Could not infer number of classes for model; provide dataset or model trained with known classes')

    model = models.resnet18(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, num_classes))
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, classes


def predict_image(model, img_pil, topk=3):
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor = test_transform(img_pil).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        topk_vals, topk_idxs = torch.topk(probs, k=min(topk, probs.size(0)))
    return topk_vals.cpu().numpy(), topk_idxs.cpu().numpy()


def main():
    st.set_page_config(page_title='Sea Classifier — Inference', layout='centered')
    st.title('Sea Animals Classifier — Inference')

    st.markdown('Upload an image (drag & drop supported) to run the ResNet18 classifier and see top predictions.')

    model_path = 'resnet18_model.pth'
    uploaded_model = None
    if not os.path.exists(model_path):
        uploaded_model = st.file_uploader('No model found. Upload a `.pth` model file', type=['pth'])
        if uploaded_model is not None:
            with open(model_path, 'wb') as f:
                f.write(uploaded_model.getbuffer())
            st.success('Saved model to resnet18_model.pth')

    try:
        model, classes = load_model(model_path)
    except Exception as e:
        st.error(f'Failed to load model: {e}')
        return

    if classes is None:
        classes = load_classes() or []

    uploaded_file = st.file_uploader('Upload image', type=['jpg', 'jpeg', 'png'])
    if uploaded_file is None:
        st.info('Choose an image to test the model.')
        return

    image = Image.open(io.BytesIO(uploaded_file.read())).convert('RGB')
    st.image(image, caption='Input image', use_column_width=True)

    if st.button('Run prediction'):
        try:
            top_vals, top_idxs = predict_image(model, image, topk=3)
            if len(classes) >= max(top_idxs) + 1:
                labels = [classes[i] for i in top_idxs]
            else:
                labels = [str(int(i)) for i in top_idxs]

            results = {lab: float(val) for lab, val in zip(labels, top_vals)}
            st.write('Top predictions:')
            for lab, val in zip(labels, top_vals):
                st.write(f'{lab}: {val*100:.2f}%')

            st.bar_chart({k: v for k, v in results.items()})
        except Exception as e:
            st.error(f'Prediction failed: {e}')


if __name__ == '__main__':
    main()
