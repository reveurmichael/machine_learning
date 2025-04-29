import streamlit as st
import torch
import torchvision
from torchvision import transforms
from torchvision.models.segmentation import fcn_resnet50
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import VOCSegmentation
import time
import requests

# Set page config
st.set_page_config(page_title="Image Segmentation App", layout="wide")

# Title and description
st.title("Image Segmentation with PyTorch")
st.write("""
## Simple Image Segmentation App
This app demonstrates how to perform semantic segmentation using PyTorch.
Semantic segmentation assigns a class label to each pixel in an image.
""")

# Sidebar options
st.sidebar.header("Options")
option = st.sidebar.selectbox(
    "Choose an action",
    ["Explain Semantic Segmentation", "Train Model", "Test on Sample Images", "Upload Your Image"]
)

# Define VOC classes
CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# Create a colormap for visualization
def create_pascal_label_colormap():
    """Creates a colormap for visualizing segmentation results."""
    colormap = np.zeros((21, 3), dtype=np.uint8)
    colormap[0] = [0, 0, 0]  # background
    colormap[1] = [128, 0, 0]  # aeroplane
    colormap[2] = [0, 128, 0]  # bicycle
    colormap[3] = [128, 128, 0]  # bird
    colormap[4] = [0, 0, 128]  # boat
    colormap[5] = [128, 0, 128]  # bottle
    colormap[6] = [0, 128, 128]  # bus
    colormap[7] = [128, 128, 128]  # car
    colormap[8] = [64, 0, 0]  # cat
    colormap[9] = [192, 0, 0]  # chair
    colormap[10] = [64, 128, 0]  # cow
    colormap[11] = [192, 128, 0]  # diningtable
    colormap[12] = [64, 0, 128]  # dog
    colormap[13] = [192, 0, 128]  # horse
    colormap[14] = [64, 128, 128]  # motorbike
    colormap[15] = [192, 128, 128]  # person
    colormap[16] = [0, 64, 0]  # pottedplant
    colormap[17] = [128, 64, 0]  # sheep
    colormap[18] = [0, 192, 0]  # sofa
    colormap[19] = [128, 192, 0]  # train
    colormap[20] = [0, 64, 128]  # tvmonitor
    return colormap

colormap = create_pascal_label_colormap()

# Function to visualize segmentation result
def decode_segmentation_masks(mask, colormap, n_classes):
    """Converts segmentation mask to a color image."""
    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    
    for i in range(n_classes):
        idx = mask == i
        r[idx] = colormap[i, 0]
        g[idx] = colormap[i, 1]
        b[idx] = colormap[i, 2]
    
    rgb = np.stack([r, g, b], axis=2)
    return rgb

# Function to load pre-trained model
@st.cache_resource
def load_model():
    model = fcn_resnet50(pretrained=True)
    model.eval()
    return model

# Function to predict on a single image
def predict(model, img):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(img)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)["out"][0]
    
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions

# Function to train a model
def train_model(data_dir, num_epochs=5):
    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    target_transform = transforms.Compose([
        transforms.Resize((256, 256), interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
    
    # Load dataset
    try:
        train_dataset = VOCSegmentation(
            root=data_dir,
            year='2012',
            image_set='train',
            download=True,
            transform=transform,
            target_transform=target_transform
        )
        
        # Split dataset
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
        
        # Load model with pretrained weights
        model = fcn_resnet50(pretrained=True)
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        loss_container = st.container()
        with loss_container:
            loss_col1, loss_col2 = st.columns(2)
            train_loss_text = loss_col1.empty()
            val_loss_text = loss_col2.empty()
        
        # Training loop
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for i, (inputs, labels) in enumerate(train_loader):
                # Convert labels to class indices
                labels = labels.squeeze(1).long()
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = model(inputs)["out"]
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                
                # Update progress
                progress = (epoch * len(train_loader) + i) / (num_epochs * len(train_loader))
                progress_bar.progress(progress)
                status_text.text(f"Training Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_loader)}")
                
                # Update loss display
                if i % 10 == 9:
                    train_loss_text.text(f"Training Loss: {running_loss/10:.4f}")
                    running_loss = 0.0
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    labels = labels.squeeze(1).long()
                    outputs = model(inputs)["out"]
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            
            val_loss_text.text(f"Validation Loss: {val_loss/len(val_loader):.4f}")
            status_text.text(f"Completed Epoch {epoch+1}/{num_epochs}")
        
        # Save the model
        torch.save(model.state_dict(), "segmentation_model.pth")
        status_text.text("Training completed and model saved!")
        
        return model
    
    except Exception as e:
        st.error(f"Error in training: {e}")
        return None

# Main content based on selected option
if option == "Step 1: Explain Semantic Segmentation":
    st.write("""
    ### What is Semantic Segmentation?
    
    Semantic segmentation is a computer vision task where we classify each pixel in an image into a specific category 
    (like person, car, tree, road, etc.). Unlike object detection that puts bounding boxes around objects, 
    segmentation gives us a pixel-by-pixel understanding of the image.
    
    ### How does it work?
    
    1. **Input**: We feed an image into a neural network.
    2. **Processing**: The network processes the image through many layers. It first extracts simple features 
       (like edges) and then more complex features (like shapes, objects) as it goes deeper.
    3. **Output**: The final output is a "mask" the same size as the input image, where each pixel is classified 
       into a category.
    
    ### Common Applications:
    - Medical imaging (identifying tumors, organs)
    - Self-driving cars (understanding road, pedestrians, other vehicles)
    - Satellite imagery analysis
    - Augmented reality effects
    
    ### The Model We're Using:
    We're using FCN (Fully Convolutional Network) with a ResNet50 backbone. This is a proven architecture that:
    - Uses a pretrained ResNet50 for feature extraction
    - Replaces fully connected layers with convolutional layers to maintain spatial information
    - Uses upsampling to restore the original image resolution
    """)

elif option == "Step 2: Train Model":
    st.write("""
    ### Train a Segmentation Model
    
    We'll train a segmentation model on the Pascal VOC 2012 dataset, which contains 20 object classes plus background.
    This is a simplified training process to demonstrate the concept.
    
    **What's happening during training:**
    1. We load the pre-trained FCN-ResNet50 model (knowledge transfer!)
    2. We fine-tune it on the Pascal VOC dataset
    3. For each image, we compare the model's prediction with the ground truth
    4. We update the model weights to improve its predictions
    
    **Technical details (simplified):**
    - We use Cross Entropy Loss to measure the difference between predictions and ground truth
    - We use Adam optimizer to update the weights
    - We train for a few epochs to keep it simple
    """)
    
    data_dir = st.text_input("Enter the directory to save dataset (leave empty for default):", "data")
    num_epochs = st.slider("Number of training epochs:", min_value=1, max_value=10, value=3)
    
    if st.button("Start Training"):
        with st.spinner("Training in progress... This might take a while!"):
            model = train_model(data_dir, num_epochs)
            if model:
                st.success("Training completed successfully!")

elif option == "Step 3: Test on Sample Images":
    st.write("""
    ### Test the Model on Sample Images
    
    Let's see how our model performs on some sample images. The model will:
    1. Analyze the image pixel by pixel
    2. Predict which class each pixel belongs to
    3. Create a color-coded mask showing the different objects
    
    The colors represent different classes (like person, car, etc.)
    """)
    
    # Load a sample image
    sample_images = [
        "https://github.com/tensorflow/models/raw/master/research/deeplab/g3doc/img/image1.jpg",
        "https://github.com/tensorflow/models/raw/master/research/deeplab/g3doc/img/image2.jpg"
    ]
    
    selected_image = st.selectbox("Choose a sample image:", ["Image 1", "Image 2"])
    image_url = sample_images[0] if selected_image == "Image 1" else sample_images[1]
    
    # Load image and display
    try:
        image = Image.open(requests.get(image_url, stream=True).raw)
        st.image(image, caption="Original Image", use_column_width=True)
        
        model = load_model()
        
        if st.button("Run Segmentation on Sample"):
            with st.spinner("Running segmentation..."):
                # Resize for processing
                original_size = image.size
                processed_image = image.resize((512, 512))
                
                # Run prediction
                segmentation_map = predict(model, processed_image)
                
                # Visualize result
                segmentation_image = decode_segmentation_masks(segmentation_map, colormap, 21)
                segmentation_image = Image.fromarray(segmentation_image)
                
                # Resize back to original size for display
                segmentation_image = segmentation_image.resize(original_size)
                
                # Display result
                st.image(segmentation_image, caption="Segmentation Result", use_column_width=True)
                
                # Show class color legend
                st.write("### Color Legend")
                cols = st.columns(7)
                for i, class_name in enumerate(CLASSES):
                    col_idx = i % 7
                    color_patch = np.ones((30, 30, 3), dtype=np.uint8) * colormap[i]
                    cols[col_idx].image(color_patch, caption=class_name, width=30)
    except Exception as e:
        st.error(f"Error processing image: {e}")

elif option == "Step 4: Upload Your Image":
    st.write("""
    ### Upload Your Own Image
    
    Upload an image and our model will segment it, identifying different objects within it.
    
    **Note:** The model works best on images similar to those in the Pascal VOC dataset, which includes:
    people, animals, vehicles, indoor objects, etc.
    """)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        model = load_model()
        
        if st.button("Run Segmentation"):
            with st.spinner("Running segmentation on your image..."):
                # Resize for processing
                original_size = image.size
                processed_image = image.resize((512, 512))
                
                # Run prediction
                segmentation_map = predict(model, processed_image)
                
                # Visualize result
                segmentation_image = decode_segmentation_masks(segmentation_map, colormap, 21)
                segmentation_image = Image.fromarray(segmentation_image)
                
                # Resize back to original size for display
                segmentation_image = segmentation_image.resize(original_size)
                
                # Display result
                st.image(segmentation_image, caption="Segmentation Result", use_column_width=True)
                
                # Show class color legend
                st.write("### Color Legend")
                cols = st.columns(7)
                for i, class_name in enumerate(CLASSES):
                    col_idx = i % 7
                    color_patch = np.ones((30, 30, 3), dtype=np.uint8) * colormap[i]
                    cols[col_idx].image(color_patch, caption=class_name, width=30)

