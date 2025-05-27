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

This project utilizes multiple datasets for comprehensive training:

### Primary Dataset - Global Wheat Detection
- **Training Images**: 3,434 wheat field images from locations around the world
- **Training Labels**: `train.csv` with bounding box annotations
- **Test Images**: Test set for evaluation (most images are hidden)
- **Sample Submission**: `sample_submission.csv` showing the required output format

### Additional Training Datasets
1. **SPIKE Dataset**: Enhanced training with positive and negative wheat samples
   - Positive samples: Images containing wheat heads
   - Negative samples: Images without wheat heads for better discrimination
2. **Wheat2017 Dataset**: Additional annotated wheat images with JSON format annotations

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
- `SPIKE Dataset/` - Additional training data with positive/negative samples
- `wheat2017/` - Additional annotated wheat dataset

### Data Columns
- `image_id` - Unique identifier matching image filenames
- `width`, `height` - Image dimensions
- `bbox` - Bounding box coordinates `[xmin, ymin, width, height]`

**Reference**: Detailed methodology available at https://arxiv.org/abs/2005.02162

## Model Architecture

### Enhanced Base Model
- **Faster R-CNN** with **ResNet-101** backbone (upgraded from ResNet-50)
- **Feature Pyramid Network (FPN)** for multi-scale detection
- Pre-trained on COCO dataset for transfer learning, then fine-tuned on wheat detection data

### Key Features
- **Mixed Precision Training (AMP)** for improved performance and memory efficiency
- **Advanced Data Augmentation** using Albumentations library with comprehensive transforms
- **Optimized NMS (Non-Maximum Suppression)** with adaptive parameter tuning
- **AdamW Optimizer** with Cosine Annealing learning rate scheduling
- **Multi-dataset Training** combining Global Wheat, SPIKE, and Wheat2017 datasets

## Technical Implementation

### Enhanced Data Preprocessing
- Image normalization using ImageNet statistics
- Comprehensive data augmentation techniques:
  - **Geometric transforms**: Horizontal/vertical flips, 90° rotation, shift-scale-rotate
  - **Cropping**: Random resized crop for better generalization
  - **Color augmentation**: HSV adjustment, brightness/contrast, RGB shift
  - **Blur and noise**: Gaussian blur, motion blur, Gaussian noise
  - **Distortion**: Elastic transform, grid distortion, optical distortion
  - **Weather effects**: Random fog, random shadow
- Bounding box format conversion with robust error handling

### Advanced Training Strategy
- **Epochs**: 25 (optimized for full dataset usage)
- **Batch Size**: 8 (with CUDA support, utilizing mixed precision benefits)
- **Learning Rate**: 1e-4 with Cosine Annealing scheduling
- **Optimizer**: AdamW with weight decay (1e-4)
- **Loss Function**: Multi-task loss (classification + localization)
- **Mixed Precision**: Automatic Mixed Precision (AMP) with GradScaler
- **Full Dataset Training**: No validation split to maximize training data

### Optimized Post-processing
- **Adaptive NMS Parameters**: 
  - Score threshold: 0.5 (configurable)
  - NMS threshold: 0.5 (configurable)
- **Multi-parameter Testing**: Automated search for optimal thresholds
- **Robust Prediction Format**: Enhanced submission format handling

### Multi-Dataset Integration
The training incorporates three datasets using `ConcatDataset`:
1. **Global Wheat Detection**: Primary competition dataset
2. **SPIKE Dataset**: Additional positive/negative samples
3. **Wheat2017**: Supplementary annotated wheat images

## File Structure

```
├── dataset/
│   ├── train/                    # Training images (from train.zip)
│   ├── test/                     # Test images (from test.zip)
│   ├── SPIKE Dataset/            # Additional training data
│   │   ├── positive/             # Positive wheat samples
│   │   └── negative/             # Negative samples
│   ├── wheat2017/                # Wheat2017 dataset
│   ├── train.csv                 # Training annotations (Kaggle format)
│   └── sample_submission.csv     # Sample submission format
├── main.py                       # Enhanced training and inference script
├── best_model_optimized.pth      # Saved model weights
└── optimized_submission.csv      # Final predictions for submission
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

The enhanced script will:
1. Load and combine multiple datasets (Global Wheat, SPIKE, Wheat2017)
2. Apply comprehensive data augmentation
3. Train the Faster R-CNN model with mixed precision and ResNet-101 backbone
4. Use AdamW optimizer with cosine annealing learning rate scheduling
5. Save the best model based on training loss
6. Generate optimized predictions on test data with NMS parameter tuning

### Key Parameters
- **Score Threshold**: 0.5 (minimum confidence for detection)
- **NMS Threshold**: 0.5 (IoU threshold for non-maximum suppression)
- **Backbone**: ResNet-101 with FPN
- **Mixed Precision**: Enabled for CUDA devices
- **Learning Rate Scheduler**: Cosine Annealing

## Results

### Enhanced Competition Performance
![alt text](image-1.png)
- **Final Public Score**: 0.7711 (significantly improved from 0.6445)
- **Final Private Score**: 0.6733 (improved from 0.5611)

### Performance Improvements Summary
The enhanced model achieved substantial improvements through:
- **+19.6%** improvement in public score (0.6445 → 0.7711)
- **+20.0%** improvement in private score (0.5611 → 0.6733)

### Output Format
The model generates predictions in the required Kaggle competition format:
- Each detection: `[confidence_score xmin ymin width height]`
- Multiple detections per image separated by spaces
- Empty string for images with no wheat heads detected

## Model Optimizations & Enhancements

### Architecture Improvements ✅
1. **Upgraded Backbone**: ResNet-101 (from ResNet-50) for better feature extraction
2. **Mixed Precision Training**: Reduces memory usage and speeds up training
3. **Enhanced FPN**: Better multi-scale feature handling

### Training Enhancements ✅
4. **Multi-Dataset Training**: Combined Global Wheat, SPIKE, and Wheat2017 datasets
5. **Advanced Data Augmentation**: Comprehensive augmentation pipeline with 10+ techniques
6. **AdamW Optimizer**: Superior convergence compared to SGD
7. **Cosine Annealing**: Improved learning rate scheduling
8. **Full Dataset Utilization**: No validation split to maximize training data

### Post-processing Improvements ✅
9. **Optimized NMS**: Adaptive parameter tuning for better precision-recall balance
10. **Robust Error Handling**: Enhanced prediction pipeline with fallback mechanisms

## Technical Specifications

### Hardware Requirements
- **GPU**: CUDA-compatible GPU recommended for training (mixed precision support)
- **Memory**: Minimum 8GB GPU memory for batch size 8
- **CPU**: Multi-core processor for data loading

### Performance Metrics
- **Training time**: Approximately 2-3 hours on modern GPU with mixed precision
- **Inference time**: ~1-2 seconds per image
- **Model size**: ~180MB (ResNet-101 backbone)
- **Final Competition Scores**: 
  - **Public**: 0.7711 (rank improvement)
  - **Private**: 0.6733 (consistent performance)
- **Evaluation Metric**: Intersection over Union (IoU) based scoring

### Dataset Statistics
- **Total Training Samples**: 3,434 (Global Wheat) + SPIKE samples + Wheat2017 samples
- **Combined Dataset Size**: Significantly larger than baseline
- **Augmentation Factor**: ~10x effective dataset size through augmentation

## Future Improvements

- **Ensemble Methods**: Combine multiple backbone architectures
- **Test-Time Augmentation (TTA)**: Further boost inference performance
- **Advanced Anchor Optimization**: Fine-tune anchor sizes and ratios
- **Cross-Validation**: Implement k-fold CV for even better generalization
- **Pseudo-Labeling**: Use high-confidence predictions on test data
- **Attention Mechanisms**: Integrate attention modules for better feature focus

## Conclusion

This enhanced implementation demonstrates significant improvements over the baseline approach through systematic optimizations in model architecture, training strategy, and data utilization. The **19.6% improvement in public score** validates the effectiveness of the multi-faceted enhancement approach.

---

*This project was developed as part of the Visual Recognition using Deep Learning course, Spring 2025. The enhanced version showcases advanced deep learning techniques for agricultural computer vision applications.*