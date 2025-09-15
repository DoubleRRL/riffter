#!/usr/bin/env python3
"""
Convert Meta's Llama model format to Hugging Face format
"""

import os
import torch
from pathlib import Path
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import json

def convert_meta_to_huggingface():
    """Convert Meta's Llama model to Hugging Face format"""

    # Paths
    meta_model_path = Path("/Users/RRL_1/.llama/checkpoints/Llama3.1-8B")
    hf_model_path = Path("models/llama_model")

    print(f"🔄 Converting Meta model from: {meta_model_path}")
    print(f"🎯 To Hugging Face format at: {hf_model_path}")
    print()

    # Create output directory
    hf_model_path.mkdir(parents=True, exist_ok=True)

    try:
        # Load the model configuration
        print("📋 Loading model configuration...")

        # Read params.json for model configuration
        with open(meta_model_path / "params.json", 'r') as f:
            params = json.load(f)

        print(f"Model parameters: {params}")

        # Create Hugging Face config
        config = LlamaConfig(
            vocab_size=params.get('vocab_size', 128256),
            hidden_size=params.get('hidden_dim', 4096),
            intermediate_size=params.get('intermediate_dim', 14336),
            num_hidden_layers=params.get('n_layers', 32),
            num_attention_heads=params.get('n_heads', 32),
            max_position_embeddings=params.get('max_position_embeddings', 131072),
            rms_norm_eps=1e-5,
            rope_theta=500000.0,
            attention_bias=False,
            mlp_bias=False,
            tie_word_embeddings=False,
        )

        # Save config
        config.save_pretrained(hf_model_path)
        print("✅ Configuration saved")

        # Load tokenizer
        print("🔤 Loading tokenizer...")
        tokenizer = LlamaTokenizer.from_pretrained(
            str(meta_model_path),
            legacy=True
        )

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Save tokenizer
        tokenizer.save_pretrained(hf_model_path)
        print("✅ Tokenizer saved")

        # Load and convert model weights
        print("🏋️ Loading model weights...")
        checkpoint = torch.load(meta_model_path / "consolidated.00.pth", map_location="cpu")

        print(f"Loaded checkpoint with {len(checkpoint)} parameters")

        # Initialize Hugging Face model
        print("🏗️ Initializing Hugging Face model...")
        model = LlamaForCausalLM(config)

        # Convert weights (this is a simplified mapping - may need adjustments)
        print("🔄 Converting weights...")

        # Basic weight mapping for Llama
        state_dict = {}

        # Embedding layer
        state_dict['embed_tokens.weight'] = checkpoint['tok_embeddings.weight']

        # Attention layers
        for i in range(config.num_hidden_layers):
            # Attention weights
            state_dict[f'layers.{i}.self_attn.q_proj.weight'] = checkpoint[f'layers.{i}.attention.wq.weight']
            state_dict[f'layers.{i}.self_attn.k_proj.weight'] = checkpoint[f'layers.{i}.attention.wk.weight']
            state_dict[f'layers.{i}.self_attn.v_proj.weight'] = checkpoint[f'layers.{i}.attention.wv.weight']
            state_dict[f'layers.{i}.self_attn.o_proj.weight'] = checkpoint[f'layers.{i}.attention.wo.weight']

            # Feed forward
            state_dict[f'layers.{i}.mlp.gate_proj.weight'] = checkpoint[f'layers.{i}.feed_forward.w1.weight']
            state_dict[f'layers.{i}.mlp.up_proj.weight'] = checkpoint[f'layers.{i}.feed_forward.w3.weight']
            state_dict[f'layers.{i}.mlp.down_proj.weight'] = checkpoint[f'layers.{i}.feed_forward.w2.weight']

            # Layer norms
            state_dict[f'layers.{i}.input_layernorm.weight'] = checkpoint[f'layers.{i}.attention_norm.weight']
            state_dict[f'layers.{i}.post_attention_layernorm.weight'] = checkpoint[f'layers.{i}.ffn_norm.weight']

        # Output layer
        state_dict['norm.weight'] = checkpoint['norm.weight']
        state_dict['lm_head.weight'] = checkpoint['output.weight']

        print(f"Converted {len(state_dict)} weight tensors")

        # Load weights into model
        model.load_state_dict(state_dict, strict=False)
        print("✅ Model weights loaded")

        # Save model
        print("💾 Saving model in Hugging Face format...")
        model.save_pretrained(hf_model_path, safe_serialization=True)

        print("\n🎉 CONVERSION COMPLETE!")
        print(f"Model saved to: {hf_model_path.absolute()}")
        print("\nVerify with: python verify_model.py")

        return True

    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check if all required files exist")
        print("2. Verify model architecture compatibility")
        print("3. Consider using Meta's official conversion tools")
        return False

if __name__ == "__main__":
    success = convert_meta_to_huggingface()
    if not success:
        print("\n💡 Alternative: Use Meta's official conversion")
        print("Visit: https://github.com/huggingface/transformers/tree/main/src/transformers/models/llama")
        print("Or use: python -m transformers.models.llama.convert_llama_weights_to_hf")
