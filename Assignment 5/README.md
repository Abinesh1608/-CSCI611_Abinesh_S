# Neural Style Transfer Assignment

Implementation of neural style transfer from Gatys et al. (2016) paper using PyTorch.

## What it does

Takes a content image and style image, combines them using VGG19 features. The result preserves the content structure but applies the artistic style.

## Files Structure

```
A04/
├── Style_Transfer_Exercise.ipynb    # Main implementation notebook
├── README.md                       # This file
├── content_image.jpg              # Content image (mountain landscape)  
└── style_image.jpg                # Style image (Van Gogh's Starry Night)
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- PIL (Pillow)
- matplotlib
- numpy
- requests

## Installation

```bash
pip install torch torchvision pillow matplotlib numpy requests
```

## How to run

Open `Style_Transfer_Exercise.ipynb` and run cells from top to bottom. It will:
- Load the mountain landscape and Van Gogh images
- Set up VGG19 model
- Run style transfer 
- Show results

### 2. Running Experiments

The notebook includes three main experiments:

#### Experiment 1: Content vs Style Weight Ratios
Tests different balances between content preservation and style application:
- `content_weight=1, style_weight=1e3`: More content preservation
- `content_weight=1, style_weight=1e6`: Balanced (default)
- `content_weight=1, style_weight=1e8`: More stylization

#### Experiment 2: Style Layer Weight Emphasis  
Tests different weighting schemes for style layers:
- **Early Layers**: Emphasizes larger style artifacts (conv1_1, conv2_1)
- **Balanced**: Equal emphasis across layers (default)
- **Late Layers**: Emphasizes finer style details (conv4_1, conv5_1)

#### Experiment 3: Learning Rate Impact
Tests different optimization speeds:
- `lr=0.001`: Conservative learning rate
- `lr=0.003`: Default learning rate  
- `lr=0.01`: Aggressive learning rate

### 3. Customizing Images

To use your own images, replace the image files:

```python
# In Cell 8, modify these lines:
content = load_image('your_content_image.jpg').to(device)
style = load_image('your_style_image.jpg', shape=content.shape[-2:]).to(device)
```

Supported formats: JPG, PNG, and most common image formats. Images are automatically resized to 400px maximum dimension for efficiency.

## Key Components

### 1. Feature Extraction (`get_features`)
- Maps VGG19 layer indices to paper nomenclature
- Extracts features from conv1_1, conv2_1, conv3_1, conv4_1, conv4_2, conv5_1

### 2. Gram Matrix Computation (`gram_matrix`)
- Computes Gram matrices for style representation
- Captures correlations between feature maps

### 3. Loss Functions
- **Content Loss**: MSE between target and content features at conv4_2
- **Style Loss**: Sum of MSE between Gram matrices across style layers  
- **Total Loss**: Weighted combination of content and style losses

### 4. Optimization
- Uses Adam optimizer to iteratively update target image
- Target initialized as copy of content image
- Typical convergence in 2000-5000 iterations

## Settings

Main parameters:
- content_weight: 1 (how much original content to keep)
- style_weight: 1e5 (how much style to apply) 
- learning_rate: 0.01 (optimization speed)
- steps: 100 (reduced for faster testing)
- image size: 200px (reduced for speed)

Layer weights for style (early layers = big patterns, late layers = details):
conv1_1: 1.0, conv2_1: 0.8, conv3_1: 0.5, conv4_1: 0.3, conv5_1: 0.1

## Expected runtime

- Main demo: ~2-5 minutes
- Each experiment: ~30-60 seconds  
- Total: ~10-15 minutes
- Works on CPU or GPU

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `max_size` parameter or use CPU
2. **Slow convergence**: Increase learning rate or iteration count
3. **Too much stylization**: Decrease `style_weight` or increase `content_weight`
4. **Too little style**: Increase `style_weight` or adjust style layer weights

### Performance Tips
- Use GPU for faster processing: `device = torch.device("cuda")`
- Reduce image size for faster experiments: `max_size=200`
- Start with fewer iterations for testing: `steps=500`

## Implementation Details

This implementation follows the Gatys et al. (2016) paper methodology:

1. **Content Representation**: Uses conv4_2 layer of VGG19
2. **Style Representation**: Uses Gram matrices from conv1_1, conv2_1, conv3_1, conv4_1, conv5_1
3. **Optimization**: L-BFGS (paper) vs Adam (this implementation) optimizer
4. **Initialization**: Target image initialized with content image

## References

- Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image style transfer using convolutional neural networks. CVPR.
- Simonyan, K., & Zisserman, A. (2014). Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556.
