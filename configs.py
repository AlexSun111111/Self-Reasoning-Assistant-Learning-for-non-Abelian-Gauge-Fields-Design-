from pydantic import BaseModel, validator
from typing import List, Optional, Union, Tuple
from enum import Enum

from non_abelian_gauge_model.imagen_non_abelian import NonAbelianGaugeModel, NonAbelianUnet, NonAbelianUnet3D, NullUnet
from non_abelian_gauge_model.trainer import NonAbelianGaugeModelTrainer
from non_abelian_gauge_model.elucidated_non_abelian import ElucidatedNonAbelianGaugeModel
from non_abelian_gauge_model.t5 import DEFAULT_T5_NAME, get_encoded_dim

# Helper functions
def ListOrTuple(inner_type):
    return Union[List[inner_type], Tuple[inner_type]]

def SingleOrList(inner_type):
    return Union[inner_type, ListOrTuple(inner_type)]

# Noise schedule enum
class NoiseSchedule(Enum):
    cosine = 'cosine'
    linear = 'linear'

class AllowExtraBaseModel(BaseModel):
    class Config:
        extra = "allow"
        use_enum_values = True

# Configuration for NullUnet
class NullUnetConfig(BaseModel):
    is_null: bool = True

    def create(self):
        return NullUnet()

# Configuration for NonAbelianUnet
class NonAbelianUnetConfig(AllowExtraBaseModel):
    dim: int
    dim_mults: ListOrTuple(int)
    text_embed_dim: int = get_encoded_dim(DEFAULT_T5_NAME)
    cond_dim: Optional[int] = None
    channels: int = 3
    attn_dim_head: int = 32
    attn_heads: int = 16

    def create(self):
        return NonAbelianUnet(**self.dict())

# Configuration for NonAbelianUnet3D
class NonAbelianUnet3DConfig(AllowExtraBaseModel):
    dim: int
    dim_mults: ListOrTuple(int)
    text_embed_dim: int = get_encoded_dim(DEFAULT_T5_NAME)
    cond_dim: Optional[int] = None
    channels: int = 3
    attn_dim_head: int = 32
    attn_heads: int = 16

    def create(self):
        return NonAbelianUnet3D(**self.dict())

# Configuration for NonAbelianGaugeModel
class NonAbelianGaugeModelConfig(AllowExtraBaseModel):
    unets: ListOrTuple(Union[NonAbelianUnetConfig, NonAbelianUnet3DConfig, NullUnetConfig])
    image_sizes: ListOrTuple(int)
    video: bool = False
    timesteps: SingleOrList(int) = 1000
    noise_schedules: SingleOrList(NoiseSchedule) = 'cosine'
    text_encoder_name: str = DEFAULT_T5_NAME
    channels: int = 3
    loss_type: str = 'l2'
    cond_drop_prob: float = 0.5

    @validator("image_sizes")
    def check_image_sizes(cls, v, values):
        if len(v) != len(values["unets"]):
            raise ValueError(f'image sizes length {len(v)} must be equivalent to the number of unets {len(values["unets"])}')
        return v

    def create(self):
        decoder_kwargs = self.dict()
        unets_kwargs = decoder_kwargs.pop('unets')
        is_video = decoder_kwargs.pop('video', False)

        unets = []

        for unet, unet_kwargs in zip(self.unets, unets_kwargs):
            if isinstance(unet, NullUnetConfig):
                unet_klass = NullUnet
            elif is_video:
                unet_klass = NonAbelianUnet3D
            else:
                unet_klass = NonAbelianUnet

            unets.append(unet_klass(**unet_kwargs))

        model = NonAbelianGaugeModel(unets, **decoder_kwargs)
        model._config = self.dict().copy()
        return model

# Configuration for ElucidatedNonAbelianGaugeModel
class ElucidatedNonAbelianGaugeModelConfig(AllowExtraBaseModel):
    unets: ListOrTuple(Union[NonAbelianUnetConfig, NonAbelianUnet3DConfig, NullUnetConfig])
    image_sizes: ListOrTuple(int)
    video: bool = False
    text_encoder_name: str = DEFAULT_T5_NAME
    channels: int = 3
    cond_drop_prob: float = 0.5
    num_sample_steps: SingleOrList(int) = 32
    sigma_min: SingleOrList(float) = 0.002
    sigma_max: SingleOrList(int) = 80
    sigma_data: SingleOrList(float) = 0.5
    rho: SingleOrList(int) = 7
    P_mean: SingleOrList(float) = -1.2
    P_std: SingleOrList(float) = 1.2
    S_churn: SingleOrList(int) = 80
    S_tmin: SingleOrList(float) = 0.05
    S_tmax: SingleOrList(int) = 50
    S_noise: SingleOrList(float) = 1.003

    @validator("image_sizes")
    def check_image_sizes(cls, v, values):
        if len(v) != len(values["unets"]):
            raise ValueError(f'image sizes length {len(v)} must be equivalent to the number of unets {len(values["unets"])}')
        return v

    def create(self):
        decoder_kwargs = self.dict()
        unets_kwargs = decoder_kwargs.pop('unets')
        is_video = decoder_kwargs.pop('video', False)

        unet_klass = NonAbelianUnet3D if is_video else NonAbelianUnet

        unets = []

        for unet, unet_kwargs in zip(self.unets, unets_kwargs):
            if isinstance(unet, NullUnetConfig):
                unet_klass = NullUnet
            elif is_video:
                unet_klass = NonAbelianUnet3D
            else:
                unet_klass = NonAbelianUnet

            unets.append(unet_klass(**unet_kwargs))

        model = ElucidatedNonAbelianGaugeModel(unets, **decoder_kwargs)
        model._config = self.dict().copy()
        return model

# Configuration for NonAbelianGaugeModelTrainer
class NonAbelianGaugeModelTrainerConfig(AllowExtraBaseModel):
    model: dict
    elucidated: bool = False
    video: bool = False
    use_ema: bool = True
    lr: SingleOrList(float) = 1e-4
    eps: SingleOrList(float) = 1e-8
    beta1: float = 0.9
    beta2: float = 0.99
    max_grad_norm: Optional[float] = None
    group_wd_params: bool = True
    warmup_steps: SingleOrList(Optional[int]) = None
    cosine_decay_max_steps: SingleOrList(Optional[int]) = None

    def create(self):
        trainer_kwargs = self.dict()

        model_config = trainer_kwargs.pop('model')
        elucidated = trainer_kwargs.pop('elucidated')
        video = trainer_kwargs.pop('video')

        model_config_klass = ElucidatedNonAbelianGaugeModelConfig if elucidated else NonAbelianGaugeModelConfig
        model = model_config_klass(**{**model_config, 'video': video}).create()

        return NonAbelianGaugeModelTrainer(model, **trainer_kwargs)
