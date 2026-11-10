# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
"""Matched public Edge release used by GISTdata training and asset preparation."""

GISTDATA_EDGE_REPOSITORY = "nvidia/Cosmos3-Edge"
GISTDATA_EDGE_REVISION = "5b590285c565623fb8a47c123708d27866b0564a"
GISTDATA_EDGE_TRAINABLE_KEYS = (
    "moe_gen",
    "k_norm_und_for_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "action_modality_embed",
)
