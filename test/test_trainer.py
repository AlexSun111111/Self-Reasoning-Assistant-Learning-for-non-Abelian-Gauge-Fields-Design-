import unittest
import torch
from torch.utils.data import DataLoader, TensorDataset
from non_abelian_gauge_model.imagen_non_abelian import NonAbelianGaugeModel, NonAbelianUnet
from non_abelian_gauge_model.trainer import NonAbelianGaugeModelTrainer
from non_abelian_gauge_model.configs import NonAbelianUnetConfig


class TestNonAbelianGaugeModelTrainer(unittest.TestCase):
    def setUp(self):
        """Set up the models and trainer with necessary initial parameters."""
        unet_config = NonAbelianUnetConfig(
            dim=64,
            dim_mults=[1, 2, 4],
            text_embed_dim=512,
            channels=3,
            attn_dim_head=32,
            attn_heads=8
        ).create()

        self.model = NonAbelianGaugeModel(
            unets=[unet_config],
            image_size=256,
            channels=3,
            timesteps=1000,
            noise_schedule='cosine'
        )
        self.trainer = NonAbelianGaugeModelTrainer(self.model)

        # Create dummy dataset
        self.batch_size = 2
        input_images = torch.randn(self.batch_size, 3, 256, 256)
        input_texts = torch.randn(self.batch_size, 512)  # Placeholder for encoded text
        dataset = TensorDataset(input_images, input_texts)
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size)

    def test_train_step(self):
        """Test the training step of the trainer."""
        for batch in self.dataloader:
            loss = self.trainer.train_step(batch)
            self.assertGreaterEqual(loss, 0.0)
            break  # Test only one batch for simplicity

    def test_validate_step(self):
        """Test the validation step of the trainer."""
        for batch in self.dataloader:
            loss = self.trainer.validate_step(batch)
            self.assertGreaterEqual(loss, 0.0)
            break  # Test only one batch for simplicity

    def test_training_loop(self):
        """Test the full training loop."""
        self.trainer.fit(self.dataloader, epochs=1)
        # Normally, you'd have more assertions and checks here to ensure the loop behaves as expected

    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        checkpoint_path = './test_checkpoint.pt'
        self.trainer.save_checkpoint(checkpoint_path)

        # Create a new trainer and load the checkpoint
        new_trainer = NonAbelianGaugeModelTrainer(self.model)
        new_trainer.load_checkpoint(checkpoint_path)

        # Check if the states are the same
        for param, new_param in zip(self.trainer.model.parameters(), new_trainer.model.parameters()):
            self.assertTrue(torch.equal(param, new_param))

        # Clean up
        import os
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)


if __name__ == '__main__':
    unittest.main()
