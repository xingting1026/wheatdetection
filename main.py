import os
import pandas as pd
import numpy as np
import cv2
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import nms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 混合精度訓練導入
from torch.cuda.amp import autocast, GradScaler

# 設定裝置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用裝置: {device}")
print(f"支持混合精度: {torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7}")

class WheatDataset(Dataset):
    def __init__(self, dataframe, image_dir, transforms=None):
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        
        # 按image_id分組，每個圖片的所有bbox
        self.image_ids = self.df['image_id'].unique()
        
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        
        # 讀取圖片
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 獲取該圖片的所有bbox
        records = self.df[self.df['image_id'] == image_id]
        
        boxes = []
        area = []
        labels = []
        
        for _, record in records.iterrows():
            if pd.notna(record['bbox']):
                # 解析bbox字串 [xmin, ymin, width, height]
                bbox = ast.literal_eval(record['bbox'])
                xmin, ymin, width, height = bbox
                xmax = xmin + width
                ymax = ymin + height
                
                boxes.append([xmin, ymin, xmax, ymax])
                area.append(width * height)
                labels.append(1)  # 小麥穗類別為1（0為背景）
        
        # 如果沒有bbox，建立空的target
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            area = torch.zeros((0,), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            area = torch.as_tensor(area, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["image_id"] = torch.tensor([index])
        
        if self.transforms:
            sample = {
                'image': image,
                'bboxes': boxes.tolist() if len(boxes) > 0 else [],
                'labels': labels.tolist() if len(labels) > 0 else []
            }
            try:
                sample = self.transforms(**sample)
                image = sample['image']
                if len(sample['bboxes']) > 0:
                    target["boxes"] = torch.as_tensor(sample['bboxes'], dtype=torch.float32)
                else:
                    target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            except Exception as e:
                # 如果轉換失敗，使用原始資料
                print(f"轉換失敗 {image_id}: {e}")
                transform_basic = A.Compose([
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2(p=1.0)
                ])
                image = transform_basic(image=image)['image']
        
        return image, target
    
    def __len__(self):
        return len(self.image_ids)

def get_transforms(train=False):
    if train:
        transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.9),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.9),
            ], p=0.9),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        transform = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    
    return transform

def get_test_transforms():
    """專門為測試資料設計的transform，不包含bbox處理"""
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(p=1.0)
    ])
    return transform

def get_model(backbone='resnet101'):
    """
    獲取模型，支持不同的backbone
    backbone: 'resnet50', 'resnet101', 'resnext101'
    """
    print(f"使用backbone: {backbone}")
    
    if backbone == 'resnet50':
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    elif backbone == 'resnet101':
        # 使用ResNet101作為backbone
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.models.detection import FasterRCNN
        
        backbone = resnet_fpn_backbone('resnet101', pretrained=True)
        model = FasterRCNN(backbone, num_classes=91)  # 先用預訓練的類別數
    else:
        raise ValueError(f"不支持的backbone: {backbone}")
    
    # 取得分類器的輸入特徵數
    num_classes = 2  # 1個類別(小麥穗) + 背景
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 替換預訓練的頭部
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def collate_fn(batch):
    return tuple(zip(*batch))

class TestDataset(Dataset):
    def __init__(self, image_ids, image_dir, transforms=None):
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transforms = transforms
    
    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented['image']
        
        return image, image_id
    
    def __len__(self):
        return len(self.image_ids)

def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler, print_freq=100):
    """
    訓練一個epoch，支持混合精度
    """
    model.train()
    running_loss = 0.0
    
    for i, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        optimizer.zero_grad()
        
        # 使用混合精度訓練
        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        
        # 混合精度反向傳播
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += losses.item()
        
        if i % print_freq == 0:
            print(f'Epoch [{epoch}], Step [{i}/{len(data_loader)}], Loss: {losses.item():.4f}')
    
    return running_loss / len(data_loader)

def apply_nms_optimization(boxes, scores, labels, score_threshold=0.5, nms_threshold=0.5):
    """
    優化的NMS後處理
    測試不同的閾值組合來找到最佳性能
    """
    # 過濾低分預測
    valid_indices = scores >= score_threshold
    valid_boxes = boxes[valid_indices]
    valid_scores = scores[valid_indices]
    valid_labels = labels[valid_indices]
    
    if len(valid_boxes) == 0:
        return torch.tensor([]), torch.tensor([]), torch.tensor([])
    
    # 應用NMS
    keep_indices = nms(valid_boxes, valid_scores, nms_threshold)
    
    final_boxes = valid_boxes[keep_indices]
    final_scores = valid_scores[keep_indices]
    final_labels = valid_labels[keep_indices]
    
    return final_boxes, final_scores, final_labels

def make_predictions_optimized(model, test_loader, device, 
                             score_threshold=0.5, nms_threshold=0.5):
    """
    優化的預測函數，包含NMS調優
    """
    predictions = []
    
    print(f"使用參數 - Score threshold: {score_threshold}, NMS threshold: {nms_threshold}")
    
    model.eval()  # 確保模型在評估模式
    with torch.no_grad():
        for images, image_ids in test_loader:
            # 確保images是list格式
            if not isinstance(images, list):
                images = list(images)
            images = [img.to(device) for img in images]
            
            # 使用混合精度推理
            with autocast():
                outputs = model(images)
            
            for i, output in enumerate(outputs):
                image_id = image_ids[i]
                
                scores = output['scores']
                boxes = output['boxes']
                labels = output['labels']
                
                # 應用優化的NMS
                final_boxes, final_scores, final_labels = apply_nms_optimization(
                    boxes, scores, labels, score_threshold, nms_threshold
                )
                
                if len(final_boxes) > 0:
                    # 轉換為提交格式 [score xmin ymin width height]
                    pred_strings = []
                    for box, score in zip(final_boxes.cpu().numpy(), final_scores.cpu().numpy()):
                        xmin, ymin, xmax, ymax = box
                        width = xmax - xmin
                        height = ymax - ymin
                        pred_strings.append(f"{score:.4f} {xmin:.0f} {ymin:.0f} {width:.0f} {height:.0f}")
                    
                    prediction_string = " ".join(pred_strings)
                else:
                    prediction_string = ""
                
                predictions.append({
                    'image_id': image_id,
                    'PredictionString': prediction_string
                })
    
    return predictions

def main():
    # 讀取資料
    print("讀取訓練資料...")
    train_df = pd.read_csv('train.csv')
    
    print(f"總資料筆數: {len(train_df)}")
    print(f"唯一圖片數: {train_df['image_id'].nunique()}")
    
    # 使用全部訓練資料 - 不分割驗證集  
    image_ids = train_df['image_id'].unique()
    print(f"使用全部訓練圖片數: {len(image_ids)}")
    
    # 建立資料集 - 使用全部數據
    train_dataset = WheatDataset(
        train_df,  # 使用完整數據集
        'train', 
        transforms=get_transforms(train=True)
    )
    
    # 增加batch size以利用混合精度的優勢
    batch_size = 8 if torch.cuda.is_available() else 4
    print(f"使用batch size: {batch_size}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows系統設為0
        collate_fn=collate_fn
    )
    
    # 建立模型 - 使用ResNet101 backbone
    print("建立模型...")
    model = get_model(backbone='resnet101')
    model.to(device)
    
    # 優化器 - 使用AdamW替代SGD
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    
    # 學習率排程器 - 使用Cosine Annealing
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    
    # 混合精度訓練的Scaler
    scaler = GradScaler()
    
    # 訓練迴圈 - 使用全部數據可以減少訓練輪數
    num_epochs = 40  # 全部數據訓練，適當減少輪數
    best_loss = float('inf')
    
    print("開始訓練...")
    for epoch in range(num_epochs):
        # 訓練
        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        
        # 更新學習率
        lr_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss:.4f}, LR: {current_lr:.6f}')
        
        # 儲存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save(model.state_dict(), 'best_model_optimized.pth')
            print(f'儲存最佳模型 (Loss: {best_loss:.4f})')
    
    print("訓練完成！")
    
    # 載入最佳模型進行預測
    print("載入最佳模型進行預測...")
    model.load_state_dict(torch.load('best_model_optimized.pth'))
    
    # 測試資料預測
    print("開始對test資料夾進行預測...")
    
    # 獲取test資料夾中的所有圖片
    test_image_dir = 'test'
    test_image_files = [f for f in os.listdir(test_image_dir) if f.endswith('.jpg')]
    test_image_ids = [f.replace('.jpg', '') for f in test_image_files]
    
    print(f"找到 {len(test_image_ids)} 張測試圖片")
    
    # 建立測試資料集
    test_dataset_full = TestDataset(
        test_image_ids,
        test_image_dir,
        transforms=get_test_transforms()  # 使用專門的測試transforms
    )
    
    test_loader_full = DataLoader(
        test_dataset_full,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda x: tuple(zip(*x))  # 簡化的collate_fn for test data
    )
    
    best_score_threshold = 0.5
    best_nms_threshold = 0.5
    
    predictions_final = make_predictions_optimized(
        model, test_loader_full, device,
        score_threshold=best_score_threshold,
        nms_threshold=best_nms_threshold
    )
    
    # 建立最終提交檔案
    submission_df_final = pd.DataFrame(predictions_final)
    submission_df_final.to_csv('optimized_submission.csv', index=False)

    print("save 'submission.csv'")
    
    # 顯示預測統計
    valid_predictions = sum(1 for p in predictions_final if p['PredictionString'] != '')
    print(f"有效預測數量: {valid_predictions}/{len(predictions_final)}")

if __name__ == "__main__":
    main()