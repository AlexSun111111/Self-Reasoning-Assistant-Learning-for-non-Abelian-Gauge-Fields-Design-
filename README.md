# Self-Reasoning Assistant Learning for Non-Abelian Gauge Fields

## Abstract
This project develops a novel self-reasoning assistant learning framework to generate non-Abelian gauge fields. It highlights the application of advanced machine learning techniques to interpret and operationalize complex physical theories.

## Introduction
Non-Abelian gauge fields play a critical role in various areas of theoretical physics. This project introduces an innovative computational framework leveraging self-reasoning assistant learning to design and simulate circuits capable of manifesting non-Abelian gauge fields, facilitating experimental investigations into their properties.

![image](https://github.com/AlexSun111111/Self-Reasoning-Assistant-Learning-for-non-Abelian-Gauge-Fields-Design-/blob/main/Logo/README.png)

## Installation
Clone the repository and install the required packages:
```bash
git clone https://github.com/AlexSun111111/Self-Reasoning-Assistant-Learning-for-non-Abelian-Gauge-Fields-Design-.git
cd Self-Reasoning-Assistant-Learning-for-non-Abelian-Gauge-Fields-Design-
pip install -r requirements.txt
```

## Usage

### Text-to-Image Generation
For easier training, you can provide text strings directly instead of calculating text encodings in advance. However, for scalability, it's advisable to precompute the text embeddings and masks.

#### Training
```python
import torch
from imagen_pytorch import Unet, Imagen

unet1 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
    layer_cross_attns = (False, True, True, True)
)

unet2 = Unet(
    dim = 32,
    cond_dim = 512,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = (2, 4, 8, 8),
    layer_attns = (False, False, False, True),
    layer_cross_attns = (False, False, False, True)
)

imagen = Imagen(
    unets = (unet1, unet2),
    image_sizes = (64, 256),
    timesteps = 1000,
    cond_drop_prob = 0.1
).cuda()

# Mock images and text embeddings
text_embeds = torch.randn(4, 256, 768).cuda()
images = torch.randn(4, 3, 256, 256).cuda()

for i in (1, 2):
    loss = imagen(images, text_embeds = text_embeds, unet_number = i)
    loss.backward()

# Sampling images based on text embeddings
images = imagen.sample(texts = [
    'a whale breaching from afar',
    'young girl blowing out candles on her birthday cake',
    'fireworks with blue and green sparkles'
], cond_scale = 3.)

images.shape # (3, 3, 256, 256)
```

The number of textual captions must match the batch size of the images if you go this route. For example:

```python
texts = [
    'a child screaming at finding a worm within a half-eaten apple',
    'lizard running across the desert on two feet',
    'waking up to a psychedelic landscape',
    'seashells sparkling in the shallow waters'
]

images = torch.randn(4, 3, 256, 256).cuda()

for i in (1, 2):
    loss = imagen(images, texts = texts, unet_number = i)
    loss.backward()
```

### Training with DataLoader
You can use the Trainer to automatically train using DataLoader instances. Ensure your DataLoader returns either images (for unconditional training) or a tuple of ('images', 'text_embeds') for text-guided generation.

#### Unconditional Training Example
```python
from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset

unet = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 1,
    layer_attns = (False, False, False, True),
    layer_cross_attns = False
)

imagen = Imagen(
    condition_on_text = False,
    unets = unet,
    image_sizes = 128,
    timesteps = 1000
)

trainer = ImagenTrainer(
    imagen = imagen,
    split_valid_from_train = True
).cuda()

dataset = Dataset('/path/to/training/images', image_size = 128)
trainer.add_train_dataset(dataset, batch_size = 16)

for i in range(200000):
    loss = trainer.train_step(unet_number = 1, max_batch_size = 4)
    print(f'loss: {loss}')

    if not (i % 50):
        valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 4)
        print(f'valid loss: {valid_loss}')

    if not (i % 100) and trainer.is_main:
        images = trainer.sample(batch_size = 1, return_pil_images = True)
        images[0].save(f'./sample-{i // 100}.png')
```

## Code Structure
Overview of the codebase:
- `simulate.py`: Main script for running simulations.
- `models/`: Contains the AI models for generating gauge fields.
- `data/`: Dataset directory, includes sample data and format specifications.

## Dataset
Details on the datasets used, including access and preprocessing steps, can be found in `Dataset Example (100 pairs)`.

## Results
Summary of the results with an emphasis on the effectiveness of the proposed framework. Refer to our paper for further look.

## Contributions
List of contributors:
- Jinyang Sun, Portland Institute, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- Xi Chen, College of Integrated Circuit Science and Engineering, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- Xiumei Wang, College of Electronic and Optical Engineering, Nanjing University of Posts and Telecommunications, Nanjing 210003, China
- Dandan Zhu, Institute of AI Education, East China Normal University, Shanghai 200333, China
- Xingping Zhou, Institute of Quantum Information and Technology, Nanjing University of Posts and Telecommunications, Nanjing 210003, China

† ddzhu@mail.ecnu.edu.cn  
‡ zxp@njupt.edu.cn  

## Citing
If you use this project or the associated paper in your research, please cite it as follows:
```bibtex
@article{
  title={Self-Reasoning Assistant Learning for Non-Abelian Gauge Fields},
  author={Jinyang Sun, Xi Chen, Xiumei Wang, Dandan Zhu and Xingping Zhou},
  journal={xx},
  year={2024},
  volume={xx},
  pages={xx-xx}
}
```

## License
Specify the license under which the project is released.

## Acknowledgments
Thanks to the funding sources, contributors, and any research institutions involved.

### Explanation:
- **Dataset Section**: Added a reference to the `Dataset Example (100 pairs)` for detailed information about the dataset.
- **General Structure**: Kept the structure consistent, ensuring all sections are clearly defined and easy to follow.

Ensure that the `Dataset Example (100 pairs)` file is well-documented and provide the necessary details about the dataset, including how it is formatted, any preprocessing steps, and examples of the data pairs. This will help users understand how to use the dataset effectively in their own experiments.

