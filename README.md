# Global Wheat Detection

**Visual Recognition using Deep Learning**  
**2025 Spring, Final Project**

## Team Members
- **313551044** 曾偉杰
- **313551078** 吳年茵  
- **110550046** 吳孟謙
- **111550029** 蔡奕庠

## Project Overview

This project implements a deep learning solution for the **Global Wheat Detection Kaggle Competition**. The challenge involves detecting and localizing wheat heads in field images collected from various locations worldwide. The goal is to accurately predict bounding boxes around wheat heads, which is crucial for agricultural monitoring, yield estimation, and crop management.

**Competition Goal**: Predict bounding boxes around wheat heads in test images. If no wheat heads are present, predict no bounding boxes.

## Dataset

This project is based on the **Global Wheat Detection** Kaggle competition dataset. The dataset contains:

### Data Structure
- **Training Images**: 3,434 wheat field images from locations around the world
- **Training Labels**: `train.csv` with bounding box annotations
- **Test Images**: Test set for evaluation (most images are hidden)
- **Sample Submission**: `sample_submission.csv` showing the required output format

### Data Format
- **Image Format**: JPG images of wheat fields
- **Bounding Boxes**: Python-style list format `[xmin, ymin, width, height]`
- **Coverage**: Not all images contain wheat heads
- **Global Scope**: Images recorded from multiple international locations

### Files Structure
- `train.csv` - Training data with bounding box annotations
- `train.zip` - Training images (643.57 MB)
- `test.zip` - Test images  
- `sample_submission.csv` - Submission format template

### Data Columns
- `image_id` - Unique identifier matching image filenames
- `width`, `height` - Image dimensions
- `bbox` - Bounding box coordinates `[xmin, ymin, width, height]`

**Reference**: Detailed methodology available at https://arxiv.org/abs/2005.02162

## Model Architecture

### Base Model
- **Faster R-CNN** with **ResNet-101** backbone
- **Feature Pyramid Network (FPN)** for multi-scale detection
- Pre-trained on COCO dataset for transfer learning, then fine-tuned on wheat detection data

### Key Features
- **Mixed Precision Training** for improved performance and memory efficiency
- **Advanced Data Augmentation** using Albumentations library
- **Optimized NMS (Non-Maximum Suppression)** for better post-processing
- **AdamW Optimizer** with Cosine Annealing learning rate scheduling

## Technical Implementation

### Data Preprocessing
- Image normalization using ImageNet statistics
- Data augmentation techniques:
  - Horizontal and vertical flips
  - Random rotation (90°)
  - Color jittering (HSV and brightness/contrast)
- Bounding box format conversion (Kaggle format to Pascal VOC for training)

### Training Strategy
- **Epochs**: 40
- **Batch Size**: 8 (with CUDA support)
- **Learning Rate**: 1e-4 with Cosine Annealing scheduling
- **Loss Function**: Multi-task loss (classification + localization)
- **Mixed Precision**: Automatic Mixed Precision (AMP) for efficiency

### Post-processing
- Configurable confidence score threshold (default: 0.5)
- Non-Maximum Suppression with IoU threshold (default: 0.5)
- Optimized prediction format for submission

## File Structure

```
├── train/                 # Training images (from train.zip)
├── test/                  # Test images (from test.zip)
├── train.csv             # Training annotations (Kaggle format)
├── sample_submission.csv # Sample submission format
├── main.py               # Main training and inference script
├── best_model_optimized.pth    # Saved model weights
└── optimized_submission.csv    # Final predictions for submission
```

## Dependencies

```python
torch>=1.9.0
torchvision>=0.10.0
opencv-python
pandas
numpy
albumentations
matplotlib
scikit-learn
```

## Usage

### Training
```bash
python main.py
```

The script will:
1. Load and preprocess the training data
2. Train the Faster R-CNN model with mixed precision
3. Save the best model based on training loss
4. Generate predictions on test data

### Key Parameters
- **Score Threshold**: 0.5 (minimum confidence for detection)
- **NMS Threshold**: 0.5 (IoU threshold for non-maximum suppression)
- **Backbone**: ResNet-101 with FPN

## Results

### Competition Performance
- **Final Score**: 0.6445
- **Private Score**: 0.5611
![Result](image.png)

### Output Format
The model generates predictions in the required Kaggle competition format:
- Each detection: `[confidence_score xmin ymin width height]`
- Multiple detections per image separated by spaces
- Empty string for images with no wheat heads detected

## Model Optimizations

1. **Mixed Precision Training**: Reduces memory usage and speeds up training
2. **Advanced Backbone**: ResNet-101 provides better feature extraction than ResNet-50
3. **Optimized Data Augmentation**: Carefully selected augmentations that preserve bounding box integrity
4. **Learning Rate Scheduling**: Cosine annealing for stable convergence
5. **NMS Optimization**: Tunable thresholds for optimal precision-recall balance

## Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended for training
- **Memory**: Minimum 8GB GPU memory for batch size 8
- **CPU**: Multi-core processor for data loading

## Performance Notes

- **Training time**: Approximately 2-3 hours on modern GPU
- **Inference time**: ~1-2 seconds per image
- **Model size**: ~160MB (ResNet-101 backbone)
- **Final Competition Score**: 0.6445 (Public) / 0.5611 (Private)
- **Evaluation Metric**: Intersection over Union (IoU) based scoring

## Future Improvements

- Ensemble methods with multiple backbones
- Test-time augmentation (TTA)
- Advanced anchor optimization
- Cross-validation for better generalization

---

*This project was developed as part of the Visual Recognition using Deep Learning course, Spring 2025.*