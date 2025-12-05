import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import time
import copy

# --- Configuration ---
DATA_DIR = '/content/drive/MyDrive/flowers_data' 
MODEL_NAME = 'densenet201'
NUM_CLASSES = 102
BATCH_SIZE = 32
NUM_EPOCHS = 20 
LEARNING_RATE = 0.001
FINE_TUNE_LR = 1e-5 
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")

# --- 1. Data Loading and Augmentation (CRUCIAL FOR ACCURACY) ---
# Define aggressive transformations for training for high accuracy
train_transforms = transforms.Compose([
    transforms.RandomRotation(30), # Increase rotational variance
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Color augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet Normalization
])

# Define standard transformations for validation/testing
valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the datasets
try:
    image_datasets = {
        'train': datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), train_transforms),
        'valid': datasets.ImageFolder(os.path.join(DATA_DIR, 'valid'), valid_transforms)
    }
except Exception as e:
    print(f"Error loading data. Check if '{DATA_DIR}' exists and contains 'train'/'valid' folders.")
    print(f"Exception: {e}")
    exit()

# Create data loaders
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'valid': DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
class_names = image_datasets['train'].classes

# --- 2. Model Initialization (Transfer Learning) ---

def initialize_model(model_name, num_classes, use_pretrained=True):
    """Initializes the DenseNet-201 model with a custom classifier."""

    # Load pre-trained weights for transfer learning
    model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)

    # Freeze all feature parameters initially (Phase 1)
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final classification layer (model.classifier)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, num_classes)
    )

    return model

# Initialize the model and move to the device
model = initialize_model(MODEL_NAME, NUM_CLASSES, use_pretrained=True)
model.to(DEVICE)
# Define Loss function
criterion = nn.CrossEntropyLoss()

# --- 3. Training Function ---

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    """
    Main training loop implementation.

    This function is structured to support a two-phase fine-tuning approach:
    1. Train only the classifier (feature extraction).
    2. Unfreeze and fine-tune the entire model (global fine-tuning).
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('-' * 10)
        print(f'Epoch {epoch+1}/{num_epochs}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                # Apply learning rate scheduler step
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                # Zero the parameter gradients
                optimizer.zero_grad()
                # Forward pass (track history only in train)
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best accuracy seen so far
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best valid Acc: {best_acc:.4f}')
    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

# --- 4. Training Execution (Two-Phase Fine-Tuning) ---

# PHASE 1: Feature Extraction (Train only the head/classifier)
print("\n" + "="*50)
print("PHASE 1: TRAINING CLASSIFIER (FEATURE EXTRACTION)")
print("="*50)

# Ensure only classifier parameters require gradients
for name, param in model.named_parameters():
    param.requires_grad = name.startswith('classifier')

# Optimize only the trainable parameters (the classifier)
optimizer_ft_1 = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
# Set up a scheduler (e.g., step down the learning rate every 7 epochs)
exp_lr_scheduler_1 = optim.lr_scheduler.StepLR(optimizer_ft_1, step_size=7, gamma=0.1)
# Train the classifier for a few epochs
model = train_model(model, criterion, optimizer_ft_1, exp_lr_scheduler_1, num_epochs=5)

# PHASE 2: Global Fine-Tuning (Train all layers with a very low LR)
print("\n" + "="*50)
print("PHASE 2: GLOBAL FINE-TUNING (TRAINING ALL LAYERS)")
print("="*50)

# Unfreeze all layers
for param in model.parameters():
    param.requires_grad = True
# Use a new optimizer for all parameters with a much lower learning rate
optimizer_ft_2 = optim.Adam(model.parameters(), lr=FINE_TUNE_LR)
# Use a learning rate scheduler for the entire fine-tuning process
exp_lr_scheduler_2 = optim.lr_scheduler.StepLR(optimizer_ft_2, step_size=5, gamma=0.1)
# Continue training for the remaining epochs
model = train_model(model, criterion, optimizer_ft_2, exp_lr_scheduler_2, num_epochs=NUM_EPOCHS - 5)

# --- 5. Model Saving ---
# Save the final best model weights
final_model_path = 'densenet201_best_model.pth'
torch.save(model.state_dict(), final_model_path)
print(f'\nFinal best model saved to {final_model_path}')