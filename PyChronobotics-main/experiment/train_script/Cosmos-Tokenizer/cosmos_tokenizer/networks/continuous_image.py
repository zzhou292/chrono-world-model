# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The continuous image tokenizer with VAE or AE formulation for 2D data."""

from collections import OrderedDict, namedtuple

import torch
from loguru import logger as logging
from torch import nn

from cosmos_tokenizer.modules import (
    ContinuousFormulation,
    DecoderType,
    EncoderType,
)

NetworkEval = namedtuple("NetworkEval", ["reconstructions", "posteriors", "latent"])


class ContinuousImageTokenizer(nn.Module):
    def __init__(
        self, z_channels: int, z_factor: int, latent_channels: int, **kwargs
    ) -> None:
        super().__init__()
        self.name = kwargs.get("name", "ContinuousImageTokenizer")
        self.latent_channels = latent_channels

        encoder_name = kwargs.get("encoder", EncoderType.Default.name)
        self.encoder = EncoderType[encoder_name].value(
            z_channels=z_factor * z_channels, **kwargs
        )

        decoder_name = kwargs.get("decoder", DecoderType.Default.name)
        self.decoder = DecoderType[decoder_name].value(z_channels=z_channels, **kwargs)

        self.quant_conv = torch.nn.Conv2d(
            z_factor * z_channels, z_factor * latent_channels, 1
        )
        self.post_quant_conv = torch.nn.Conv2d(latent_channels, z_channels, 1)

        formulation_name = kwargs.get("formulation", ContinuousFormulation.AE.name)
        self.distribution = ContinuousFormulation[formulation_name].value()
        logging.info(
            f"{self.name} based on {formulation_name} formulation, with {kwargs}."
        )

        num_parameters = sum(param.numel() for param in self.parameters())
        logging.info(f"model={self.name}, num_parameters={num_parameters:,}")
        logging.info(
            f"z_channels={z_channels}, latent_channels={self.latent_channels}."
        )

    def encoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("encoder", self.encoder),
                    ("quant_conv", self.quant_conv),
                    ("distribution", self.distribution),
                ]
            )
        )

    def decoder_jit(self):
        return nn.Sequential(
            OrderedDict(
                [
                    ("post_quant_conv", self.post_quant_conv),
                    ("decoder", self.decoder),
                ]
            )
        )

    def last_decoder_layer(self):
        return self.decoder.conv_out

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        return self.distribution(moments)

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input) -> dict[str, torch.Tensor] | NetworkEval:
        latent, posteriors = self.encode(input)
        dec = self.decode(latent)
        if self.training:
            return dict(reconstructions=dec, posteriors=posteriors, latent=latent)
        return NetworkEval(reconstructions=dec, posteriors=posteriors, latent=latent)
