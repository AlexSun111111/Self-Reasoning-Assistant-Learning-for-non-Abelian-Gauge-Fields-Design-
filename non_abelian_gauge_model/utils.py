import torch
import json
from pathlib import Path

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def safe_get(d, keys, default=None):
    """Safely get a nested dictionary value."""
    for key in keys:
        d = d.get(key, default)
        if d is default:
            return default
    return d

def load_non_abelian_model_from_checkpoint(checkpoint_path, model_class, *model_args, **model_kwargs):
    """Load a model from a checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    assert checkpoint_path.exists(), f'Checkpoint not found at {checkpoint_path.resolve()}'

    # Initialize the model
    model = model_class(*model_args, **model_kwargs)

    # Load the state dictionary
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict['model_state_dict'])
    return model

def custom_slugify(text, max_length=255):
    """Slugify a text string."""
    return text.replace('-', '_').replace(',', '').replace(' ', '_').replace('|', '--').strip('-_./\\')[:max_length]

def save_config(config, path):
    """Save a configuration to a JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(path):
    """Load a configuration from a JSON file."""
    with open(path, 'r') as f:
        config = json.load(f)
    return config

# Placeholder functions for model-specific utilities

def preprocess_image(image, image_size):
    """Preprocess an image (placeholder)."""
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
    ])
    return transform(image)

def postprocess_output(output):
    """Postprocess model output (placeholder)."""
    # Implement postprocessing if needed
    return output

# Example usage of the utilities
if __name__ == "__main__":
    # Example usage for saving and loading a config
    config = {
        'model': {
            'unet_config': {
                'dim': 64,
                'dim_mults': [1, 2, 4, 8],
                'text_embed_dim': 512,
            },
            'image_size': 256,
            'channels': 3,
            'timesteps': 1000,
            'noise_schedule': 'cosine',
        },
        'training': {
            'lr': 1e-4,
            'batch_size': 16,
            'epochs': 100,
        }
    }
    save_config(config, 'config.json')
    loaded_config = load_config('config.json')
    print(loaded_config)
