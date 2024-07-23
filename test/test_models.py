import unittest
import torch
from non_abelian_gauge_model.imagen_non_abelian import NonAbelianGaugeModel, NonAbelianUnet, NonAbelianUnet3D, NullUnet
from non_abelian_gauge_model.trainer import NonAbelianGaugeModelTrainer
from non_abelian_gauge_model.elucidated_non_abelian import ElucidatedNonAbelianGaugeModel
from non_abelian_gauge_model.configs import NonAbelianUnetConfig, NonAbelianGaugeModelConfig, ElucidatedNonAbelianGaugeModelConfig

class TestNonAbelianGaugeModel(unittest.TestCase):
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

    def test_model_output(self):
        """Test if model output matches expected output."""
        input_image = torch.randn(1, 3, 256, 256)
        t = torch.randint(0, 1000, (1,))
        output = self.model(input_image, t)
        self.assertEqual(output.shape, input_image.shape)

    def test_training_step(self):
        """Test the training step of the model."""
        input_image = torch.randn(1, 3, 256, 256)
        input_text = torch.randn(1, 512)  # Placeholder for encoded text
        batch = (input_image, input_text)
        loss = self.trainer.train_step(batch)
        self.assertGreaterEqual(loss, 0.0)

class TestElucidatedNonAbelianGaugeModel(unittest.TestCase):
    def setUp(self):
        """Set up the elucidated model with necessary initial parameters."""
        unet_config = NonAbelianUnetConfig(
            dim=64,
            dim_mults=[1, 2, 4],
            text_embed_dim=512,
            channels=3,
            attn_dim_head=32,
            attn_heads=8
        ).create()

        self.elucidated_model = ElucidatedNonAbelianGaugeModel(
            unets=[unet_config],
            image_size=256,
            channels=3,
            cond_drop_prob=0.5,
            num_sample_steps=32,
            sigma_min=0.002,
            sigma_max=80,
            sigma_data=0.5,
            rho=7,
            P_mean=-1.2,
            P_std=1.2,
            S_churn=80,
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003
        )

    def test_elucidated_model_output(self):
        """Test if elucidated model output matches expected output."""
        input_image = torch.randn(1, 3, 256, 256)
        t = torch.randint(0, 1000, (1,))
        output = self.elucidated_model(input_image, t)
        self.assertEqual(output.shape, input_image.shape)

    def test_elucidated_model_training_step(self):
        """Test the training step of the elucidated model."""
        optimizer = torch.optim.Adam(self.elucidated_model.parameters(), lr=1e-4)
        input_image = torch.randn(1, 3, 256, 256)
        input_text = torch.randn(1, 512)  # Placeholder for encoded text
        batch = (input_image, input_text)
        loss = self.elucidated_model.training_step(batch, optimizer)
        self.assertGreaterEqual(loss, 0.0)

class TestConfigurations(unittest.TestCase):
    def test_non_abelian_unet_config(self):
        """Test NonAbelianUnet configuration creation."""
        config = NonAbelianUnetConfig(
            dim=64,
            dim_mults=[1, 2, 4],
            text_embed_dim=512,
            channels=3,
            attn_dim_head=32,
            attn_heads=8
        )
        unet = config.create()
        self.assertIsInstance(unet, NonAbelianUnet)

    def test_non_abelian_gauge_model_config(self):
        """Test NonAbelianGaugeModel configuration creation."""
        unet_config = NonAbelianUnetConfig(
            dim=64,
            dim_mults=[1, 2, 4],
            text_embed_dim=512,
            channels=3,
            attn_dim_head=32,
            attn_heads=8
        )
        config = NonAbelianGaugeModelConfig(
            unets=[unet_config],
            image_sizes=[256],
            timesteps=1000,
            noise_schedules='cosine'
        )
        model = config.create()
        self.assertIsInstance(model, NonAbelianGaugeModel)

    def test_elucidated_non_abelian_gauge_model_config(self):
        """Test ElucidatedNonAbelianGaugeModel configuration creation."""
        unet_config = NonAbelianUnetConfig(
            dim=64,
            dim_mults=[1, 2, 4],
            text_embed_dim=512,
            channels=3,
            attn_dim_head=32,
            attn_heads=8
        )
        config = ElucidatedNonAbelianGaugeModelConfig(
            unets=[unet_config],
            image_sizes=[256],
            cond_drop_prob=0.5,
            num_sample_steps=32,
            sigma_min=0.002,
            sigma_max=80,
            sigma_data=0.5,
            rho=7,
            P_mean=-1.2,
            P_std=1.2,
            S_churn=80,
            S_tmin=0.05,
            S_tmax=50,
            S_noise=1.003
        )
        model = config.create()
        self.assertIsInstance(model, ElucidatedNonAbelianGaugeModel)

if __name__ == '__main__':
    unittest.main()

