import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

class NonAbelianGaugeModelTrainer:
    def __init__(self, model, lr=1e-4, betas=(0.9, 0.99), eps=1e-8, weight_decay=0, max_grad_norm=None, use_ema=True):
        self.model = model
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.use_ema = use_ema

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=self.betas, eps=self.eps, weight_decay=self.weight_decay)
        self.ema_model = None
        if self.use_ema:
            self.ema_model = self.initialize_ema_model()

    def initialize_ema_model(self):
        ema_model = NonAbelianGaugeModel(
            unets=self.model.unets,
            image_size=self.model.image_size,
            channels=self.model.channels,
            timesteps=self.model.timesteps,
            noise_schedule=self.model.noise_schedule
        )
        ema_model.load_state_dict(self.model.state_dict())
        return ema_model

    def update_ema(self, beta=0.999):
        if self.ema_model is None:
            return
        with torch.no_grad():
            ema_params = self.ema_model.parameters()
            model_params = self.model.parameters()
            for ema_param, model_param in zip(ema_params, model_params):
                ema_param.data.mul_(beta).add_(model_param.data, alpha=1 - beta)

    def train_step(self, batch):
        self.model.train()
        images, texts = batch
        images, texts = images.cuda(), texts.cuda()
        self.optimizer.zero_grad()
        loss = self.compute_loss(images, texts)
        loss.backward()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.use_ema:
            self.update_ema()
        return loss.item()

    def compute_loss(self, images, texts):
        # Define your loss function here
        # Example: loss = nn.MSELoss()(self.model(images), texts)
        outputs = self.model(images, texts)
        loss = nn.MSELoss()(outputs, images)  # Placeholder, replace with appropriate loss calculation
        return loss

    def validate_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            images, texts = batch
            images, texts = images.cuda(), texts.cuda()
            loss = self.compute_loss(images, texts)
        return loss.item()

    def fit(self, train_loader, val_loader=None, epochs=100, log_every=10, validate_every=10, save_every=10, checkpoint_path='./checkpoint.pt'):
        for epoch in range(epochs):
            total_loss = 0
            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):
                loss = self.train_step(batch)
                total_loss += loss

            avg_loss = total_loss / len(train_loader)
            if epoch % log_every == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

            if val_loader and epoch % validate_every == 0:
                val_loss = self.evaluate(val_loader)
                print(f"Validation Loss: {val_loss:.4f}")

            if epoch % save_every == 0:
                self.save_checkpoint(checkpoint_path)

    def evaluate(self, val_loader):
        total_loss = 0
        for batch in tqdm(val_loader, desc="Validating"):
            loss = self.validate_step(batch)
            total_loss += loss
        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def save_checkpoint(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        if self.use_ema:
            checkpoint['ema_model_state_dict'] = self.ema_model.state_dict()
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.use_ema and 'ema_model_state_dict' in checkpoint:
            self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
        print(f"Checkpoint loaded from {path}")
