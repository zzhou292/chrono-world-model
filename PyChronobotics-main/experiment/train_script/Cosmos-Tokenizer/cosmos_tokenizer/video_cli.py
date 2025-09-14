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
"""A CLI to run CausalVideoTokenizer on plain videos based on torch.jit.

Usage:
    python3 -m cosmos_tokenizer.video_cli \
        --video_pattern 'path/to/video/samples/*.mp4' \
        --output_dir ./reconstructions \
        --checkpoint_enc ./pretrained_ckpts/CosmosCV_f4x8x8/encoder.jit \
        --checkpoint_dec ./pretrained_ckpts/CosmosCV_f4x8x8/decoder.jit

    Optionally, you can run the model in pure PyTorch mode:
    python3 -m cosmos_tokenizer.video_cli \
        --video_pattern 'path/to/video/samples/*.mp4' \
        --mode=torch \
        --tokenizer_type=CV \
        --temporal_compression=4 \
        --spatial_compression=8 \
        --checkpoint_enc ./pretrained_ckpts/CosmosCV_f4x8x8/encoder.jit \
        --checkpoint_dec ./pretrained_ckpts/CosmosCV_f4x8x8/decoder.jit
"""

import os
from argparse import ArgumentParser, Namespace
from typing import Any
import sys

import numpy as np
from loguru import logger as logging

from cosmos_tokenizer.networks import TokenizerConfigs
from cosmos_tokenizer.utils import (
    get_filepaths,
    get_output_filepath,
    read_video,
    resize_video,
    write_video,
)
from cosmos_tokenizer.video_lib import CausalVideoTokenizer


def _parse_args() -> tuple[Namespace, dict[str, Any]]:
    parser = ArgumentParser(description="A CLI for CausalVideoTokenizer.")
    parser.add_argument(
        "--video_pattern",
        type=str,
        default="path/to/videos/*.mp4",
        help="Glob pattern.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="JIT full Autoencoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_enc",
        type=str,
        default=None,
        help="JIT Encoder model filepath.",
    )
    parser.add_argument(
        "--checkpoint_dec",
        type=str,
        default=None,
        help="JIT Decoder model filepath.",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        choices=["CV", "DV"],
        help="Specifies the tokenizer type.",
    )
    parser.add_argument(
        "--spatial_compression",
        type=int,
        choices=[8, 16],
        default=8,
        help="The spatial compression factor.",
    )
    parser.add_argument(
        "--temporal_compression",
        type=int,
        choices=[4, 8],
        default=4,
        help="The temporal compression factor.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["torch", "jit"],
        default="jit",
        help="Specify the backend: native 'torch' or 'jit' (default: 'jit')",
    )
    parser.add_argument(
        "--short_size",
        type=int,
        default=None,
        help="The size to resample inputs. None, by default.",
    )
    parser.add_argument(
        "--temporal_window",
        type=int,
        default=17,
        help="The temporal window to operate at a time.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        help="Sets the precision, default bfloat16.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for invoking the model.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Output directory."
    )
    parser.add_argument(
        "--output_fps",
        type=float,
        default=24.0,
        help="Output frames-per-second (FPS).",
    )
    parser.add_argument(
        "--save_input",
        action="store_true",
        help="If on, the input video will be be outputted too.",
    )

    args = parser.parse_args()
    return args


logging.info("Initializes args ...")
args = _parse_args()
if args.mode == "torch" and args.tokenizer_type not in ["CV", "DV"]:
    logging.error("'torch' backend requires the tokenizer_type of 'CV' or 'DV'.")
    sys.exit(1)


def _run_eval() -> None:
    """Invokes JIT-compiled CausalVideoTokenizer on an input video."""

    if (
        args.checkpoint_enc is None
        and args.checkpoint_dec is None
        and args.checkpoint is None
    ):
        logging.warning(
            "Aborting. Both encoder or decoder JIT required. Or provide the full autoencoder JIT model."
        )
        return

    if args.mode == "torch":
        tokenizer_config = TokenizerConfigs[args.tokenizer_type].value
        tokenizer_config.update(dict(spatial_compression=args.spatial_compression))
        tokenizer_config.update(dict(temporal_compression=args.temporal_compression))
    else:
        tokenizer_config = None

    logging.info(
        f"Loading a torch.jit model `{os.path.dirname(args.checkpoint or args.checkpoint_enc or args.checkpoint_dec)}` ..."
    )
    autoencoder = CausalVideoTokenizer(
        checkpoint=args.checkpoint,
        checkpoint_enc=args.checkpoint_enc,
        checkpoint_dec=args.checkpoint_dec,
        tokenizer_config=tokenizer_config,
        device=args.device,
        dtype=args.dtype,
    )

    logging.info(f"Looking for files matching video_pattern={args.video_pattern} ...")
    filepaths = get_filepaths(args.video_pattern)
    logging.info(f"Found {len(filepaths)} videos from {args.video_pattern}.")

    for filepath in filepaths:
        logging.info(f"Reading video {filepath} ...")
        video = read_video(filepath)
        video = resize_video(video, short_size=args.short_size)

        logging.info("Invoking the autoencoder model in ... ")
        batch_video = video[np.newaxis, ...]
        output_video = autoencoder(batch_video, temporal_window=args.temporal_window)[0]
        logging.info("Constructing output filepath ...")
        output_filepath = get_output_filepath(filepath, output_dir=args.output_dir)
        logging.info(f"Outputing {output_filepath} ...")
        write_video(output_filepath, output_video, fps=args.output_fps)
        if args.save_input:
            ext = os.path.splitext(output_filepath)[-1]
            input_filepath = output_filepath.replace(ext, "_input" + ext)
            write_video(input_filepath, video, fps=args.output_fps)


@logging.catch(reraise=True)
def main() -> None:
    _run_eval()


if __name__ == "__main__":
    main()
