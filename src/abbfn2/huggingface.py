# Copyright 2025 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Any

import jax.numpy as jnp
import jax
from jax.random import PRNGKey
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.traverse_util import flatten_dict, unflatten_dict
from omegaconf import DictConfig
from transformers import FlaxPreTrainedModel, PretrainedConfig

from abbfn2.bfn.output_network.model import BFNMultimodalOutput
from abbfn2.bfn.types import OutputNetworkPredictionMM, ThetaMM
from abbfn2.bfn.factory import get_bfn


class BFNMultimodalConfig(PretrainedConfig):
    """Configuration for the HuggingFace-compatible Multimodal BFN."""

    model_type = "abbfn2_multimodal_bfn"

    def __init__(
        self,
        bfn_cfgs: dict[str, DictConfig] | None = None,
        network_cfg: DictConfig | None = None,
        **kwargs,
    ):
        """
        Initialises the BFNMultimodalConfig.

        Args:
            bfn_cfgs: A dictionary where keys are data mode names and values are
                DictConfigs for instantiating each BFNBase object using `get_bfn`.
            network_cfg: The DictConfig for the BFNMultimodalOutput network.
            **kwargs: Additional keyword arguments passed to the PretrainedConfig parent class.
        """
        super().__init__(**kwargs)
        self.bfn_cfgs = bfn_cfgs
        self.network_cfg = network_cfg


class HFBFN(FlaxPreTrainedModel):
    """HuggingFace-compatible wrapper for MultimodalBFN."""

    config_class = BFNMultimodalConfig
    base_model_prefix = "bfn"

    def __init__(
        self,
        config: BFNMultimodalConfig,
        module: nn.Module = BFNMultimodalOutput,
        seed: int = 0,
        dtype: jnp.dtype = jnp.float32,
        _do_init: bool = True,
    ):
        self.bfn = get_bfn(
            DictConfig(config.bfn_cfgs),
            DictConfig(config.network_cfg),
        )
        self.bfns = self.bfn.bfns
        self.data_modes = self.bfn.data_modes
        self.output_network_cfg = self.bfn.output_network_cfg
        self._split_key_for_dms = self.bfn._split_key_for_dms
        self._replicate_across_dms = self.bfn._replicate_across_dms

        super().__init__(config, module, seed=seed, _do_init=_do_init)


    def init_weights(
        self, rng: jax.random.PRNGKey, input_shape: Tuple, params: FrozenDict | None = None
    ) -> FrozenDict:
        """
        Initializes the model weights. This method is called by FlaxPreTrainedModel constructor and `from_pretrained`.
        """
        random_params = self.bfn.init(key=rng)

        if params is not None:
            random_params = flatten_dict(unfreeze(random_params))
            params = flatten_dict(unfreeze(params))
            for missing_key in self._missing_keys:
                params[missing_key] = random_params[missing_key]
            self._missing_keys = set()
            return freeze(unflatten_dict(params))
        else:
            return random_params

    def get_prior_input_distribution(self) -> ThetaMM:
        """Proxies to `MultimodalBFN.get_prior_input_distribution`."""
        return self.bfn.get_prior_input_distribution()

    def apply_output_network(
        self,
        params: dict[str, Any],
        theta: ThetaMM,
        t: float,
    ) -> OutputNetworkPredictionMM:
        """Proxies to `MultimodalBFN.apply_output_network`."""
        return self.bfn.apply_output_network(params, theta, t)

    def compute_beta(self, t: float) -> jax.Array:
        """Proxies to `MultimodalBFN.compute_beta`."""
        return self.bfn.compute_beta(t)

    def sample_sender_distribution(
        self,
        x: dict[str, jax.Array | None],
        alpha: dict[str, jax.Array],
        key: PRNGKey,
    ) -> dict[str, jax.Array | None]:
        """Proxies to `MultimodalBFN.sample_sender_distribution`."""
        return self.bfn.sample_sender_distribution(x, alpha, key)

    def sample_receiver_distribution(
        self,
        pred: OutputNetworkPredictionMM,
        alpha: dict[str, jax.Array],
        key: PRNGKey,
    ) -> dict[str, jax.Array]:
        """Proxies to `MultimodalBFN.sample_receiver_distribution`."""
        return self.bfn.sample_receiver_distribution(pred, alpha, key)

    def update_distribution(
        self,
        theta: ThetaMM,
        y: dict[str, jax.Array],
        alpha: dict[str, jax.Array | None],
        conditional_score: dict[str, jax.Array | None] | None = None,
        conditional_mask: dict[str, jax.Array | None] | None = None,
    ) -> ThetaMM:
        """Proxies to `MultimodalBFN.update_distribution`."""
        return self.bfn.update_distribution(
            theta, y, alpha, conditional_score, conditional_mask
        )

    def conditional_log_prob(
        self,
        pred: OutputNetworkPredictionMM,
        x: dict[str, jax.Array | None],
        mask: dict[str, jax.Array | None],
        theta: ThetaMM,
    ) -> float:
        """Proxies to `MultimodalBFN.conditional_log_prob`."""
        return self.bfn.conditional_log_prob(pred, x, mask, theta)
