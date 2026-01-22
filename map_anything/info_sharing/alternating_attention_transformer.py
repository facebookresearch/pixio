"""
UniCeption Alternating-Attention Transformer for Information Sharing
"""

from functools import partial
from typing import Callable, List, Optional, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from map_anything.info_sharing.base import (
    MultiViewTransformerInput,
    MultiViewTransformerOutput,
)
from uniception.models.utils.intermediate_feature_return import IntermediateFeatureReturner, feature_take_indices
from uniception.models.utils.positional_encoding import PositionGetter
from uniception.models.utils.transformer_blocks import Mlp, SelfAttentionBlock


class MultiViewAlternatingAttentionTransformer(nn.Module):
    "UniCeption Multi-View Alternating-Attention Transformer for information sharing across image features from different views."

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        distinguish_ref_and_non_ref_views: bool = True,
        use_pe_for_non_reference_views: bool = False,
        max_num_views_for_pe: int = 1000,
        use_rand_idx_pe_for_non_reference_views: bool = True,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: Type[nn.Module] = nn.GELU,
        norm_layer: Union[Type[nn.Module], Callable[..., nn.Module]] = partial(nn.LayerNorm, eps=1e-6),
        mlp_layer: Type[nn.Module] = Mlp,
        custom_positional_encoding: Optional[Callable] = None,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
        pretrained_checkpoint_path: Optional[str] = None,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the Multi-View Alternating-Attention Transformer for information sharing across image features from different views.
        Alternates between global and frame-level attention.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            distinguish_ref_and_non_ref_views (bool): Whether to identify/distinguish reference and non-reference views by using positional encoding. (default: True)
            use_pe_for_non_reference_views (bool): Whether to use view positional encoding for input non-reference views. Only used if distinguish_ref_and_non_ref_views flag is True. (default: False)
            max_num_views_for_pe (int): Maximum number of views for positional encoding. (default: 1000)
            use_rand_idx_pe_for_non_reference_views (bool): Whether to use random index positional encoding for non-reference views. (default: True)
            size (str): String to indicate interpretable size of the transformer (for e.g., base, large, ...). (default: None)
            depth (int): Number of transformer layers. (default: 12, base size)
            dim (int): Dimension of the transformer. (default: 768, base size)
            num_heads (int): Number of attention heads. (default: 12, base size)
            mlp_ratio (float): Ratio of hidden to input dimension in MLP (default: 4.)
            qkv_bias (bool): Whether to include bias in qkv projection (default: True)
            qk_norm (bool): Whether to normalize q and k (default: False)
            proj_drop (float): Dropout rate for output (default: 0.)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            init_values (float): Initial value for LayerScale gamma (default: None)
            drop_path (float): Dropout rate for stochastic depth (default: 0.)
            act_layer (nn.Module): Activation layer (default: nn.GELU)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            mlp_layer (nn.Module): MLP layer (default: Mlp)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
            use_scalable_softmax (bool): Whether to use scalable softmax (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. (default: None)
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing for memory efficiency. (default: False)
        """
        # Initialize the base class
        super().__init__(*args, **kwargs)
        self.name = name
        self.size = size

        # Initialize the specific attributes of the transformer
        self.input_embed_dim = input_embed_dim
        self.distinguish_ref_and_non_ref_views = distinguish_ref_and_non_ref_views
        self.use_pe_for_non_reference_views = use_pe_for_non_reference_views
        self.max_num_views_for_pe = max_num_views_for_pe
        self.use_rand_idx_pe_for_non_reference_views = use_rand_idx_pe_for_non_reference_views
        self.depth = depth
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop
        self.init_values = init_values
        self.drop_path = drop_path
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.mlp_layer = mlp_layer
        self.custom_positional_encoding = custom_positional_encoding
        self.use_scalable_softmax = use_scalable_softmax
        self.use_entropy_scaling = use_entropy_scaling
        self.base_token_count_for_entropy_scaling = base_token_count_for_entropy_scaling
        self.entropy_scaling_growth_factor = entropy_scaling_growth_factor
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.gradient_checkpointing = gradient_checkpointing

        # Initialize the projection layer for input embeddings
        if self.input_embed_dim != self.dim:
            self.proj_embed = nn.Linear(self.input_embed_dim, self.dim, bias=True)
        else:
            self.proj_embed = nn.Identity()

        # Initialize the self-attention blocks which ingest all views at once
        self.self_attention_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=self.dim,
                    num_heads=self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=self.qkv_bias,
                    qk_norm=self.qk_norm,
                    proj_drop=self.proj_drop,
                    attn_drop=self.attn_drop,
                    init_values=self.init_values,
                    drop_path=self.drop_path,
                    act_layer=self.act_layer,
                    norm_layer=self.norm_layer,
                    mlp_layer=self.mlp_layer,
                    custom_positional_encoding=self.custom_positional_encoding,
                    use_scalable_softmax=self.use_scalable_softmax,
                    use_entropy_scaling=self.use_entropy_scaling,
                    base_token_count_for_entropy_scaling=self.base_token_count_for_entropy_scaling,
                    entropy_scaling_growth_factor=self.entropy_scaling_growth_factor,
                )
                for _ in range(self.depth)
            ]
        )

        # Initialize the final normalization layer
        self.norm = self.norm_layer(self.dim)

        # Initialize the position getter for patch positions if required
        if self.custom_positional_encoding is not None:
            self.position_getter = PositionGetter()

        # Initialize the positional encoding table for the views if required
        if self.distinguish_ref_and_non_ref_views:
            if self.use_pe_for_non_reference_views:
                # Initialize the positional encoding table for the different views
                self.register_buffer(
                    "view_pos_table",
                    self._get_sinusoid_encoding_table(self.max_num_views_for_pe, self.dim, 10000),
                )
            else:
                # Initialize the positional encoding table for the reference view
                self.register_buffer(
                    "view_pos_table",
                    self._get_sinusoid_encoding_table(1, self.dim, 10000),
                )

        # Initialize random weights
        self.initialize_weights()

        # Apply gradient checkpointing if enabled
        if self.gradient_checkpointing:
            for i, block in enumerate(self.self_attention_blocks):
                self.self_attention_blocks[i] = self.wrap_module_with_gradient_checkpointing(block)

        # Load pretrained weights if provided
        if self.pretrained_checkpoint_path is not None:
            print(
                f"Loading pretrained multi-view Alternating-Attention transformer weights from {self.pretrained_checkpoint_path} ..."
            )
            ckpt = torch.load(self.pretrained_checkpoint_path, weights_only=False)
            print(self.load_state_dict(ckpt["model"]))
    
    def wrap_module_with_gradient_checkpointing(self, module: nn.Module):
        """
        Wrapper for Gradient Checkpointing
        """

        class _CheckpointingWrapper(module.__class__):
            _restore_cls = module.__class__

            def forward(self, *args, **kwargs):
                return checkpoint(super().forward, *args, use_reentrant=False, **kwargs)

        module.__class__ = _CheckpointingWrapper
        return module

    def _get_sinusoid_encoding_table(self, n_position, d_hid, base):
        "Sinusoid position encoding table"

        def get_position_angle_vec(position):
            return [position / np.power(base, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table)

    def initialize_weights(self):
        "Initialize weights of the transformer."
        # Linears and layer norms
        self.apply(self._init_weights)

    def _init_weights(self, m):
        "Initialize the transformer linear and layer norm weights."
        if isinstance(m, nn.Linear):
            # We use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> MultiViewTransformerOutput:
        """
        Forward interface for the Multi-View Alternating-Attention Transformer.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.
                Optionally, the input can also include additional_input_tokens (e.g., class token, registers, pose tokens, scale token)
                which are appended to the token set from the multi-view features. The tokens are of size (batch, input_embed_dim, num_of_additional_tokens).

        Returns:
            MultiViewTransformerOutput: Output of the model post information sharing.
        """
        # Check that the number of views matches the input and the features are of expected shape
        if self.use_pe_for_non_reference_views:
            assert (
                len(model_input.features) <= self.max_num_views_for_pe
            ), f"Expected less than {self.max_num_views_for_pe} views, got {len(model_input.features)}"
        assert all(
            view_features.shape[1] == self.input_embed_dim for view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            view_features.ndim == 4 for view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Initialize the multi-view features from the model input and number of views for current input
        multi_view_features = model_input.features
        num_of_views = len(multi_view_features)
        batch_size, _, height, width = multi_view_features[0].shape
        num_of_tokens_per_view = height * width

        # Stack the multi-view features (N, C, H, W) to (N, V, C, H, W) (assumes all V views have same shape)
        multi_view_features = torch.stack(multi_view_features, dim=1)

        # Resize the multi-view features from NVCHW to NLC, where L = V * H * W
        multi_view_features = multi_view_features.permute(0, 1, 3, 4, 2)  # (N, V, H, W, C)
        multi_view_features = multi_view_features.reshape(
            batch_size, num_of_views * height * width, self.input_embed_dim
        ).contiguous()

        # Process additional input tokens if provided
        if model_input.additional_input_tokens is not None:

            additional_tokens = model_input.additional_input_tokens
            assert additional_tokens.ndim == 3, "Additional tokens must have 3 dimensions (N, C, T)"
            assert (
                additional_tokens.shape[1] == self.input_embed_dim
            ), f"Additional tokens must have input dimension {self.input_embed_dim}"
            assert additional_tokens.shape[0] == batch_size, "Batch size mismatch for additional tokens"

            # Reshape to channel-last format for transformer processing
            additional_tokens = additional_tokens.permute(0, 2, 1).contiguous()  # (N, C, T) -> (N, T, C)

            # Concatenate the additional tokens to the multi-view features
            multi_view_features = torch.cat([multi_view_features, additional_tokens], dim=1)

        # Project input features to the transformer dimension
        multi_view_features = self.proj_embed(multi_view_features)

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, multi_view_features.device)
            ] * num_of_views  # List of length V, where each tensor is (N, H * W, C)
            multi_view_positions = torch.cat(multi_view_positions, dim=1)  # (N, V * H * W, C)
        else:
            multi_view_positions = [None] * num_of_views

        # Add None positions for additional tokens if they exist
        if model_input.additional_input_tokens is not None:

            additional_tokens_positions = [None] * model_input.additional_input_tokens.shape[1]
            multi_view_positions = multi_view_positions + additional_tokens_positions

        if self.distinguish_ref_and_non_ref_views:
            # Add positional encoding for reference view (idx 0)
            ref_view_pe = self.view_pos_table[0].clone().detach()
            ref_view_pe = ref_view_pe.reshape((1, 1, self.dim))
            ref_view_pe = ref_view_pe.repeat(batch_size, num_of_tokens_per_view, 1)
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]
            ref_view_features = ref_view_features + ref_view_pe
        else:
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]

        if self.distinguish_ref_and_non_ref_views and self.use_pe_for_non_reference_views:
            # Add positional encoding for non-reference views (sequential indices starting from idx 1 or random indices which are uniformly sampled)
            if self.use_rand_idx_pe_for_non_reference_views:
                non_ref_view_pe_indices = torch.randint(low=1, high=self.max_num_views_for_pe, size=(num_of_views - 1,))
            else:
                non_ref_view_pe_indices = torch.arange(1, num_of_views)
            non_ref_view_pe = self.view_pos_table[non_ref_view_pe_indices].clone().detach()
            non_ref_view_pe = non_ref_view_pe.reshape((1, num_of_views - 1, self.dim))
            non_ref_view_pe = non_ref_view_pe.repeat_interleave(num_of_tokens_per_view, dim=1)
            non_ref_view_pe = non_ref_view_pe.repeat(batch_size, 1, 1)
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]
            non_ref_view_features = non_ref_view_features + non_ref_view_pe
        else:
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]

        # Concatenate the reference and non-reference view features
        # Handle additional tokens (no view-based positional encoding for them)
        if model_input.additional_input_tokens is not None:

            additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features, additional_features], dim=1)
        else:
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features], dim=1)

        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            if depth_idx % 2 == 0:
                # Apply the self-attention block and update the multi-view features
                # Global attention across all views
                multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)
            else:
                # Handle additional tokens separately for frame-level attention
                additional_features = None
                additional_positions = None
                if model_input.additional_input_tokens is not None:

                    # Extract additional token features
                    additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
                    # Keep only view features for frame-level attention
                    multi_view_features = multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

                    # Handle positions for additional tokens if custom positional encoding is used
                    if self.custom_positional_encoding is not None:
                        additional_positions = multi_view_positions[:, num_of_views * num_of_tokens_per_view :, :]
                        multi_view_positions = multi_view_positions[:, : num_of_views * num_of_tokens_per_view, :]

                # Reshape the multi-view features from (N, V * H * W, C) to (N * V, H * W, C)
                multi_view_features = multi_view_features.reshape(
                    batch_size * num_of_views, num_of_tokens_per_view, self.dim
                ).contiguous()  # (N * V, H * W, C)
                if multi_view_positions[0] is not None:
                    multi_view_positions = multi_view_positions.reshape(
                        batch_size * num_of_views, num_of_tokens_per_view, 2
                    ).contiguous()  # (N * V, H * W, C)

                # Apply the self-attention block and update the multi-view features
                # Frame-level attention within each view
                multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)

                # Reshape the multi-view features from (N * V, H * W, C) back to (N, V * H * W, C)
                multi_view_features = multi_view_features.reshape(
                    batch_size, num_of_views * num_of_tokens_per_view, self.dim
                ).contiguous()  # (N, V * H * W, C)
                if multi_view_positions[0] is not None:
                    multi_view_positions = multi_view_positions.reshape(
                        batch_size, num_of_views * num_of_tokens_per_view, 2
                    ).contiguous()  # (N, V * H * W, C)

                # Reattach additional tokens if they exist
                if additional_features is not None:
                    multi_view_features = torch.cat([multi_view_features, additional_features], dim=1)
                    # Reattach positions for additional tokens if they exist
                    if additional_positions is not None:
                        multi_view_positions = torch.cat([multi_view_positions, additional_positions], dim=1)

        # Normalize the output features
        output_multi_view_features = self.norm(multi_view_features)

        # Extract only the view features (excluding additional tokens)
        view_features = output_multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

        # Reshape the output multi-view features (N, V * H * W, C) back to (N, V, C, H, W)
        view_features = view_features.reshape(batch_size, num_of_views, height, width, self.dim)  # (N, V, H, W, C)
        view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

        # Split the output multi-view features into separate views
        view_features = view_features.split(1, dim=1)
        view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

        # Extract and return additional token features if provided
        if model_input.additional_input_tokens is not None:

            additional_token_features = output_multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)
            return MultiViewTransformerOutput(
                features=view_features, additional_token_features=additional_token_features
            )
        else:
            return MultiViewTransformerOutput(features=view_features)


class MultiViewAlternatingAttentionTransformerIFR(
    MultiViewAlternatingAttentionTransformer, IntermediateFeatureReturner
):
    "Intermediate Feature Returner for UniCeption Multi-View Alternating-Attention Transformer"

    def __init__(
        self,
        name: str,
        input_embed_dim: int,
        distinguish_ref_and_non_ref_views: bool = True,
        use_pe_for_non_reference_views: bool = False,
        max_num_views_for_pe: int = 1000,
        use_rand_idx_pe_for_non_reference_views: bool = True,
        size: Optional[str] = None,
        depth: int = 12,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = partial(nn.LayerNorm, eps=1e-6),
        mlp_layer: nn.Module = Mlp,
        custom_positional_encoding: Callable = None,
        use_scalable_softmax: bool = False,
        use_entropy_scaling: bool = False,
        base_token_count_for_entropy_scaling: int = 444,
        entropy_scaling_growth_factor: float = 1.4,
        pretrained_checkpoint_path: str = None,
        indices: Optional[Union[int, List[int]]] = None,
        norm_intermediate: bool = True,
        intermediates_only: bool = False,
        gradient_checkpointing: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the Multi-View Alternating-Attention Transformer for information sharing across image features from different views.
        Extends the base class to return intermediate features.

        Args:
            input_embed_dim (int): Dimension of input embeddings.
            distinguish_ref_and_non_ref_views (bool): Whether to identify/distinguish reference and non-reference views by using positional encoding. (default: True)
            use_pe_for_non_reference_views (bool): Whether to use view positional encoding for input non-reference views. Only used if distinguish_ref_and_non_ref_views flag is True. (default: False)
            max_num_views_for_pe (int): Maximum number of views for positional encoding. (default: 1000)
            use_rand_idx_pe_for_non_reference_views (bool): Whether to use random index positional encoding for non-reference views. (default: True)
            size (str): String to indicate interpretable size of the transformer (for e.g., base, large, ...). (default: None)
            depth (int): Number of transformer layers. (default: 12, base size)
            dim (int): Dimension of the transformer. (default: 768, base size)
            num_heads (int): Number of attention heads. (default: 12, base size)
            mlp_ratio (float): Ratio of hidden to input dimension in MLP (default: 4.)
            qkv_bias (bool): Whether to include bias in qkv projection (default: False)
            qk_norm (bool): Whether to normalize q and k (default: False)
            proj_drop (float): Dropout rate for output (default: 0.)
            attn_drop (float): Dropout rate for attention weights (default: 0.)
            init_values (float): Initial value for LayerScale gamma (default: None)
            drop_path (float): Dropout rate for stochastic depth (default: 0.)
            act_layer (nn.Module): Activation layer (default: nn.GELU)
            norm_layer (nn.Module): Normalization layer (default: nn.LayerNorm)
            mlp_layer (nn.Module): MLP layer (default: Mlp)
            custom_positional_encoding (Callable): Custom positional encoding function (default: None)
            use_scalable_softmax (bool): Whether to use scalable softmax (default: False)
            use_entropy_scaling (bool): Whether to use entropy scaling (default: False)
            base_token_count_for_entropy_scaling (int): Base token count for entropy scaling (default: 444)
                                                        Computed using (518, 168) as base resolution with 14 patch size
            entropy_scaling_growth_factor (float): Growth factor for entropy scaling (default: 1.4)
            pretrained_checkpoint_path (str, optional): Path to the pretrained checkpoint. (default: None)
            indices (Optional[Union[int, List[int]]], optional): Indices of the layers to return. (default: None) Options:
            - None: Return all intermediate layers.
            - int: Return the last n layers.
            - List[int]: Return the intermediate layers at the specified indices.
            norm_intermediate (bool, optional): Whether to normalize the intermediate features. (default: True)
            intermediates_only (bool, optional): Whether to return only the intermediate features. (default: False)
            gradient_checkpointing (bool, optional): Whether to use gradient checkpointing for memory efficiency. (default: False)
        """
        # Init the base classes
        MultiViewAlternatingAttentionTransformer.__init__(
            self,
            name=name,
            input_embed_dim=input_embed_dim,
            distinguish_ref_and_non_ref_views=distinguish_ref_and_non_ref_views,
            use_pe_for_non_reference_views=use_pe_for_non_reference_views,
            max_num_views_for_pe=max_num_views_for_pe,
            use_rand_idx_pe_for_non_reference_views=use_rand_idx_pe_for_non_reference_views,
            size=size,
            depth=depth,
            dim=dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
            mlp_layer=mlp_layer,
            custom_positional_encoding=custom_positional_encoding,
            use_scalable_softmax=use_scalable_softmax,
            use_entropy_scaling=use_entropy_scaling,
            base_token_count_for_entropy_scaling=base_token_count_for_entropy_scaling,
            entropy_scaling_growth_factor=entropy_scaling_growth_factor,
            pretrained_checkpoint_path=pretrained_checkpoint_path,
            gradient_checkpointing=gradient_checkpointing,
            *args,
            **kwargs,
        )
        IntermediateFeatureReturner.__init__(
            self,
            indices=indices,
            norm_intermediate=norm_intermediate,
            intermediates_only=intermediates_only,
        )

    def forward(
        self,
        model_input: MultiViewTransformerInput,
    ) -> Union[
        List[MultiViewTransformerOutput],
        Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]],
    ]:
        """
        Forward interface for the Multi-View Alternating-Attention Transformer with Intermediate Feature Return.

        Args:
            model_input (MultiViewTransformerInput): Input to the model.
                Expects the features to be a list of size (batch, input_embed_dim, height, width),
                where each entry corresponds to a different view.
                Optionally, the input can also include additional_input_tokens (e.g., class token, registers, pose tokens, scale token)
                which are appended to the token set from the multi-view features. The tokens are of size (batch, input_embed_dim, num_of_additional_tokens).

        Returns:
            Union[List[MultiViewTransformerOutput], Tuple[MultiViewTransformerOutput, List[MultiViewTransformerOutput]]]:
                Output of the model post information sharing.
                If intermediates_only is True, returns a list of intermediate outputs.
                If intermediates_only is False, returns a tuple of final output and a list of intermediate outputs.
        """
        # Check that the number of views matches the input and the features are of expected shape
        if self.use_pe_for_non_reference_views:
            assert (
                len(model_input.features) <= self.max_num_views_for_pe
            ), f"Expected less than {self.max_num_views_for_pe} views, got {len(model_input.features)}"
        assert all(
            view_features.shape[1] == self.input_embed_dim for view_features in model_input.features
        ), f"All views must have input dimension {self.input_embed_dim}"
        assert all(
            view_features.ndim == 4 for view_features in model_input.features
        ), "All views must have 4 dimensions (N, C, H, W)"

        # Get the indices of the intermediate features to return
        intermediate_multi_view_features = []
        take_indices, _ = feature_take_indices(self.depth, self.indices)

        # Initialize the multi-view features from the model input and number of views for current input
        multi_view_features = model_input.features
        num_of_views = len(multi_view_features)
        batch_size, _, height, width = multi_view_features[0].shape
        num_of_tokens_per_view = height * width

        # Stack the multi-view features (N, C, H, W) to (N, V, C, H, W) (assumes all V views have same shape)
        multi_view_features = torch.stack(multi_view_features, dim=1)

        # Resize the multi-view features from NVCHW to NLC, where L = V * H * W
        multi_view_features = multi_view_features.permute(0, 1, 3, 4, 2)  # (N, V, H, W, C)
        multi_view_features = multi_view_features.reshape(
            batch_size, num_of_views * height * width, self.input_embed_dim
        ).contiguous()

        # Process additional input tokens if provided
        if model_input.additional_input_tokens is not None:

            additional_tokens = model_input.additional_input_tokens
            assert additional_tokens.ndim == 3, "Additional tokens must have 3 dimensions (N, C, T)"
            assert (
                additional_tokens.shape[1] == self.input_embed_dim
            ), f"Additional tokens must have input dimension {self.input_embed_dim}"
            assert additional_tokens.shape[0] == batch_size, "Batch size mismatch for additional tokens"

            # Reshape to channel-last format for transformer processing
            additional_tokens = additional_tokens.permute(0, 2, 1).contiguous()  # (N, C, T) -> (N, T, C)

            # Concatenate the additional tokens to the multi-view features
            multi_view_features = torch.cat([multi_view_features, additional_tokens], dim=1)

        # Project input features to the transformer dimension
        multi_view_features = self.proj_embed(multi_view_features)

        # Create patch positions for each view if custom positional encoding is used
        if self.custom_positional_encoding is not None:
            multi_view_positions = [
                self.position_getter(batch_size, height, width, multi_view_features.device)
            ] * num_of_views  # List of length V, where each tensor is (N, H * W, C)
            multi_view_positions = torch.cat(multi_view_positions, dim=1)  # (N, V * H * W, C)
        else:
            multi_view_positions = [None] * num_of_views

        # Add None positions for additional tokens if they exist
        if model_input.additional_input_tokens is not None:

            additional_tokens_positions = [None] * model_input.additional_input_tokens.shape[1]
            multi_view_positions = multi_view_positions + additional_tokens_positions

        if self.distinguish_ref_and_non_ref_views:
            # Add positional encoding for reference view (idx 0)
            ref_view_pe = self.view_pos_table[0].clone().detach()
            ref_view_pe = ref_view_pe.reshape((1, 1, self.dim))
            ref_view_pe = ref_view_pe.repeat(batch_size, num_of_tokens_per_view, 1)
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]
            ref_view_features = ref_view_features + ref_view_pe
        else:
            ref_view_features = multi_view_features[:, :num_of_tokens_per_view, :]

        if self.distinguish_ref_and_non_ref_views and self.use_pe_for_non_reference_views:
            # Add positional encoding for non-reference views (sequential indices starting from idx 1 or random indices which are uniformly sampled)
            if self.use_rand_idx_pe_for_non_reference_views:
                non_ref_view_pe_indices = torch.randint(low=1, high=self.max_num_views_for_pe, size=(num_of_views - 1,))
            else:
                non_ref_view_pe_indices = torch.arange(1, num_of_views)
            non_ref_view_pe = self.view_pos_table[non_ref_view_pe_indices].clone().detach()
            non_ref_view_pe = non_ref_view_pe.reshape((1, num_of_views - 1, self.dim))
            non_ref_view_pe = non_ref_view_pe.repeat_interleave(num_of_tokens_per_view, dim=1)
            non_ref_view_pe = non_ref_view_pe.repeat(batch_size, 1, 1)
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]
            non_ref_view_features = non_ref_view_features + non_ref_view_pe
        else:
            non_ref_view_features = multi_view_features[
                :, num_of_tokens_per_view : num_of_views * num_of_tokens_per_view, :
            ]

        # Concatenate the reference and non-reference view features
        # Handle additional tokens (no view-based positional encoding for them)
        if model_input.additional_input_tokens is not None:

            additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features, additional_features], dim=1)
        else:
            multi_view_features = torch.cat([ref_view_features, non_ref_view_features], dim=1)

        # Loop over the depth of the transformer
        for depth_idx in range(self.depth):
            if depth_idx % 2 == 0:
                # Apply the self-attention block and update the multi-view features
                # Global attention across all views
                multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)
            else:
                # Handle additional tokens separately for frame-level attention
                additional_features = None
                additional_positions = None
                if model_input.additional_input_tokens is not None:

                    # Extract additional token features
                    additional_features = multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
                    # Keep only view features for frame-level attention
                    multi_view_features = multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]

                    # Handle positions for additional tokens if custom positional encoding is used
                    if self.custom_positional_encoding is not None:
                        additional_positions = multi_view_positions[:, num_of_views * num_of_tokens_per_view :, :]
                        multi_view_positions = multi_view_positions[:, : num_of_views * num_of_tokens_per_view, :]

                # Reshape the multi-view features from (N, V * H * W, C) to (N * V, H * W, C)
                multi_view_features = multi_view_features.reshape(
                    batch_size * num_of_views, num_of_tokens_per_view, self.dim
                ).contiguous()  # (N * V, H * W, C)
                if multi_view_positions[0] is not None:
                    multi_view_positions = multi_view_positions.reshape(
                        batch_size * num_of_views, num_of_tokens_per_view, 2
                    ).contiguous()  # (N * V, H * W, C)

                # Apply the self-attention block and update the multi-view features
                # Frame-level attention within each view
                multi_view_features = self.self_attention_blocks[depth_idx](multi_view_features, multi_view_positions)

                # Reshape the multi-view features from (N * V, H * W, C) back to (N, V * H * W, C)
                multi_view_features = multi_view_features.reshape(
                    batch_size, num_of_views * num_of_tokens_per_view, self.dim
                ).contiguous()  # (N, V * H * W, C)
                if multi_view_positions[0] is not None:
                    multi_view_positions = multi_view_positions.reshape(
                        batch_size, num_of_views * num_of_tokens_per_view, 2
                    ).contiguous()  # (N, V * H * W, C)

                # Reattach additional tokens if they exist
                if additional_features is not None:
                    multi_view_features = torch.cat([multi_view_features, additional_features], dim=1)
                    # Reattach positions for additional tokens if they exist
                    if additional_positions is not None:
                        multi_view_positions = torch.cat([multi_view_positions, additional_positions], dim=1)
            if depth_idx in take_indices:
                # Normalize the intermediate features with final norm layer if enabled
                intermediate_multi_view_features.append(
                    self.norm(multi_view_features) if self.norm_intermediate else multi_view_features
                )

        # Reshape the intermediate features and convert to MultiViewTransformerOutput class
        for idx in range(len(intermediate_multi_view_features)):
            # Get the current intermediate features
            current_features = intermediate_multi_view_features[idx]

            # Extract additional token features if provided
            additional_token_features = None
            if model_input.additional_input_tokens is not None:

                additional_token_features = current_features[:, num_of_views * num_of_tokens_per_view :, :]
                additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)
                # Only keep the view features for reshaping
                current_features = current_features[:, : num_of_views * num_of_tokens_per_view, :]

            # Reshape the intermediate multi-view features (N, V * H * W, C) back to (N, V, C, H, W)
            current_features = current_features.reshape(
                batch_size, num_of_views, height, width, self.dim
            )  # (N, V, H, W, C)
            current_features = current_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

            # Split the intermediate multi-view features into separate views
            current_features = current_features.split(1, dim=1)
            current_features = [
                intermediate_view_features.squeeze(dim=1) for intermediate_view_features in current_features
            ]

            intermediate_multi_view_features[idx] = MultiViewTransformerOutput(
                features=current_features, additional_token_features=additional_token_features
            )

        # Return only the intermediate features if enabled
        if self.intermediates_only:
            return intermediate_multi_view_features

        # Normalize the output features
        output_multi_view_features = self.norm(multi_view_features)

        # Extract view features (excluding additional tokens)
        additional_token_features = None
        if model_input.additional_input_tokens is not None:

            additional_token_features = output_multi_view_features[:, num_of_views * num_of_tokens_per_view :, :]
            additional_token_features = additional_token_features.permute(0, 2, 1).contiguous()  # (N, C, T)
            view_features = output_multi_view_features[:, : num_of_views * num_of_tokens_per_view, :]
        else:
            view_features = output_multi_view_features

        # Reshape the output multi-view features (N, V * H * W, C) back to (N, V, C, H, W)
        view_features = view_features.reshape(batch_size, num_of_views, height, width, self.dim)  # (N, V, H, W, C)
        view_features = view_features.permute(0, 1, 4, 2, 3).contiguous()  # (N, V, C, H, W)

        # Split the output multi-view features into separate views
        view_features = view_features.split(1, dim=1)
        view_features = [output_view_features.squeeze(dim=1) for output_view_features in view_features]

        output_multi_view_features = MultiViewTransformerOutput(
            features=view_features, additional_token_features=additional_token_features
        )

        return output_multi_view_features, intermediate_multi_view_features


def dummy_positional_encoding(x, xpos):
    "Dummy function for positional encoding of tokens"
    x = x
    xpos = xpos
    return x


def test_reshape_for_frame_attention():
    "Test the reshape function for frame-level attention in the Alternating Attention Transformer"
    batch_size = 2
    num_of_views = 3
    height = width = 2
    dim = 4
    num_of_tokens_per_view = height * width

    # Create tensor with recognizable pattern
    x = torch.zeros(batch_size, num_of_views * num_of_tokens_per_view, dim)
    for b in range(batch_size):
        for v in range(num_of_views):
            for h in range(height):
                for w in range(width):
                    token_idx = v * num_of_tokens_per_view + h * width + w
                    x[b, token_idx] = torch.tensor([b, v, h, w])

    # Apply reshape
    reshaped = x.reshape(batch_size * num_of_views, num_of_tokens_per_view, dim).contiguous()

    # Verify shape
    assert reshaped.shape == (batch_size * num_of_views, num_of_tokens_per_view, dim)

    # Verify content (check a few values)
    for b in range(batch_size):
        for v in range(num_of_views):
            for h in range(height):
                for w in range(width):
                    batch_view_idx = b * num_of_views + v
                    token_idx = h * width + w
                    expected = torch.tensor([b, v, h, w])
                    assert torch.all(reshaped[batch_view_idx, token_idx] == expected)

    # Verify reshape back works
    back_to_original = reshaped.reshape(batch_size, num_of_views * num_of_tokens_per_view, dim)
    assert torch.all(x == back_to_original)

    print("Reshape test passed!")


if __name__ == "__main__":
    # Unit test the reshape logic used for frame-level attention
    test_reshape_for_frame_attention()

    # Init multi-view alternating-attention transformer with no custom positional encoding and run a forward pass
    for num_views in [2, 3, 4]:
        print(f"Testing MultiViewAlternatingAttentionTransformer with {num_views} views ...")
        # No positional encoding for non-reference views
        model = MultiViewAlternatingAttentionTransformer(
            name="MV-AAT",
            input_embed_dim=1024,
        )
        model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
        model_input = MultiViewTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_views
        assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)
        # Sequential idx based positional encoding
        model = MultiViewAlternatingAttentionTransformer(
            name="MV-AAT",
            input_embed_dim=1024,
            use_pe_for_non_reference_views=True,
            max_num_views_for_pe=1000,
            use_rand_idx_pe_for_non_reference_views=False,
        )
        model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
        model_input = MultiViewTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_views
        assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)
        # Random idx based positional encoding
        model = MultiViewAlternatingAttentionTransformer(
            name="MV-AAT",
            input_embed_dim=1024,
            use_pe_for_non_reference_views=True,
            max_num_views_for_pe=1000,
            use_rand_idx_pe_for_non_reference_views=True,
        )
        model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
        model_input = MultiViewTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_views
        assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)

    # Init multi-view alternating-attention transformer with custom positional encoding and run a forward pass
    for num_views in [2, 3, 4]:
        print(
            f"Testing MultiViewAlternatingAttentionTransformer with {num_views} views and custom positional encoding ..."
        )
        model = MultiViewAlternatingAttentionTransformer(
            name="MV-AAT",
            input_embed_dim=1024,
            custom_positional_encoding=dummy_positional_encoding,
        )
        model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
        model_input = MultiViewTransformerInput(features=model_input)
        model_output = model(model_input)
        assert len(model_output.features) == num_views
        assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)

    print("All multi-view alternating-attention transformers initialized and tested successfully!")

    # Intermediate Feature Returner Tests
    print("Running Intermediate Feature Returner Tests ...")

    # Run the intermediate feature returner with last-n index
    model_intermediate_feature_returner = MultiViewAlternatingAttentionTransformerIFR(
        name="MV-AAT-IFR",
        input_embed_dim=1024,
        indices=6,  # Last 6 layers
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    assert isinstance(output, tuple)
    assert isinstance(output[0], MultiViewTransformerOutput)
    assert len(output[1]) == 6
    assert all(isinstance(intermediate, MultiViewTransformerOutput) for intermediate in output[1])
    assert len(output[1][0].features) == 2

    # Run the intermediate feature returner with specific indices
    model_intermediate_feature_returner = MultiViewAlternatingAttentionTransformerIFR(
        name="MV-AAT-IFR",
        input_embed_dim=1024,
        indices=[0, 2, 4, 6],  # Specific indices
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    assert isinstance(output, tuple)
    assert isinstance(output[0], MultiViewTransformerOutput)
    assert len(output[1]) == 4
    assert all(isinstance(intermediate, MultiViewTransformerOutput) for intermediate in output[1])
    assert len(output[1][0].features) == 2

    # Test the normalizing of intermediate features
    model_intermediate_feature_returner = MultiViewAlternatingAttentionTransformerIFR(
        name="MV-AAT-IFR",
        input_embed_dim=1024,
        indices=[-1],  # Last layer
        norm_intermediate=False,  # Disable normalization
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    for view_idx in range(2):
        assert not torch.equal(
            output[0].features[view_idx], output[1][-1].features[view_idx]
        ), "Final features and intermediate features (last layer) must be different."

    model_intermediate_feature_returner = MultiViewAlternatingAttentionTransformerIFR(
        name="MV-AAT-IFR",
        input_embed_dim=1024,
        indices=[-1],  # Last layer
        norm_intermediate=True,
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(2)]
    model_input = MultiViewTransformerInput(features=model_input)
    output = model_intermediate_feature_returner(model_input)
    for view_idx in range(2):
        assert torch.equal(
            output[0].features[view_idx], output[1][-1].features[view_idx]
        ), "Final features and intermediate features (last layer) must be same."

    print("All Intermediate Feature Returner Tests passed!")

    # Test additonal input tokens for MultiViewAlternatingAttentionTransformer
    print("Testing MultiViewAlternatingAttentionTransformer with additional input tokens ...")
    model = MultiViewAlternatingAttentionTransformer(
        name="MV-AAT",
        input_embed_dim=1024,
    )
    num_views = 2
    num_additional_tokens = 5
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
    additional_tokens = torch.rand(1, 1024, num_additional_tokens)
    model_input = MultiViewTransformerInput(features=model_input, additional_input_tokens=additional_tokens)
    model_output = model(model_input)
    assert len(model_output.features) == num_views
    assert all(f.shape == (1, model.dim, 14, 14) for f in model_output.features)
    assert model_output.additional_token_features is not None
    assert model_output.additional_token_features.shape == (1, model.dim, num_additional_tokens)

    # Test additonal input tokens for MultiViewAlternatingAttentionTransformerIFR
    print("Testing MultiViewAlternatingAttentionTransformerIFR with additional input tokens ...")
    model_ifr = MultiViewAlternatingAttentionTransformerIFR(
        name="MV-AAT-IFR",
        input_embed_dim=1024,
        indices=[0, 2, 4],
    )
    model_input = [torch.rand(1, 1024, 14, 14) for _ in range(num_views)]
    additional_tokens = torch.rand(1, 1024, num_additional_tokens)
    model_input = MultiViewTransformerInput(features=model_input, additional_input_tokens=additional_tokens)
    output = model_ifr(model_input)
    assert isinstance(output, tuple)
    assert isinstance(output[0], MultiViewTransformerOutput)
    assert output[0].additional_token_features is not None
    assert output[0].additional_token_features.shape == (1, model_ifr.dim, num_additional_tokens)
    assert len(output[1]) == 3
    assert all(isinstance(intermediate, MultiViewTransformerOutput) for intermediate in output[1])
    assert all(intermediate.additional_token_features is not None for intermediate in output[1])
    assert all(
        intermediate.additional_token_features.shape == (1, model_ifr.dim, num_additional_tokens)
        for intermediate in output[1]
    )

    print("All tests using additional input tokens passed!")
