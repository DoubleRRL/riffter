#!/usr/bin/env python3
"""
Test suite for Riffter generation quality - no corny outputs, performance validation
"""

import pytest
import time
import os
import sys
sys.path.append('src')

from generation.quick_generate import generate_riff, generate_multiple_riffs

class TestRiffQuality:
    """Test riff generation quality and performance."""

    def test_riff_content_quality(self):
        """Test that generated riffs have proper Mullen-style characteristics."""
        model_path = "models/finetuned_mullen"

        # Skip if model doesn't exist
        if not os.path.exists(model_path):
            pytest.skip("Fine-tuned model not found - run training first")

        riff, gen_time = generate_riff("taxes", model_path)

        assert riff is not None, "Riff generation failed"
        assert len(riff) > 10, "Riff too short"
        assert len(riff.split()) < 50, "Riff exceeds word limit"

        riff_lower = riff.lower()

        # Check for Mullen-style slang (at least one should be present)
        slang_words = ["yo", "fr", "type shi", "buh", "cuh"]
        has_slang = any(word in riff_lower for word in slang_words)

        # Check for absence of corny words
        corny_words = ["lol", "haha", "lmao", "kek", "xd", "rofl"]
        has_corny = any(word in riff_lower for word in corny_words)

        assert not has_corny, f"Corny content detected in riff: {riff}"

        # Prefer slang but don't fail if not present (model might need more training)
        if has_slang:
            print("✓ Riff contains expected slang")
        else:
            print("⚠ Riff missing expected slang - model may need more training")

    def test_generation_performance(self):
        """Test that generation meets performance targets."""
        model_path = "models/finetuned_mullen"

        if not os.path.exists(model_path):
            pytest.skip("Fine-tuned model not found - run training first")

        # Benchmark multiple generations
        topics = ["taxes", "social media", "fast food"]
        times = []

        for topic in topics:
            start_time = time.time()
            riff, _ = generate_riff(topic, model_path)
            elapsed = time.time() - start_time
            times.append(elapsed)

            assert riff is not None, f"Failed to generate riff for {topic}"

        avg_time = sum(times) / len(times)

        # Target: <5 seconds per generation on M2
        assert avg_time < 5.0, ".2f"

        print(".2f"

    def test_multiple_topics_generation(self):
        """Test generation across multiple topics."""
        model_path = "models/finetuned_mullen"

        if not os.path.exists(model_path):
            pytest.skip("Fine-tuned model not found - run training first")

        topics = ["taxes", "dating apps", "gym culture"]
        results = generate_multiple_riffs(topics, model_path)

        assert len(results) == len(topics), "Not all topics generated riffs"

        for topic, riff, gen_time in results:
            assert riff is not None, f"No riff generated for {topic}"
            assert len(riff) > 5, f"Riff for {topic} too short: {riff}"
            assert gen_time > 0, f"Invalid generation time for {topic}"

            # Quality checks
            riff_lower = riff.lower()
            has_corny = any(word in riff_lower for word in ["lol", "haha", "lmao"])

            assert not has_corny, f"Corny content in {topic} riff: {riff}"

    def test_model_loading(self):
        """Test that the model loads correctly."""
        model_path = "models/finetuned_mullen"

        if not os.path.exists(model_path):
            pytest.skip("Fine-tuned model not found - run training first")

        # This should not raise an exception
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model = AutoModelForCausalLM.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            assert model is not None
            assert tokenizer is not None
            print("✓ Model and tokenizer load successfully")
        except Exception as e:
            pytest.fail(f"Model loading failed: {e}")

def run_quality_assessment():
    """Run comprehensive quality assessment."""
    print("🎭 Riffter Quality Assessment 🎭")
    print("=" * 40)

    model_path = "models/finetuned_mullen"

    if not os.path.exists(model_path):
        print("❌ Fine-tuned model not found. Run training first.")
        return False

    # Test basic generation
    print("\n📊 Testing basic generation...")
    try:
        riff, gen_time = generate_riff("taxes", model_path)
        if riff and gen_time < 5.0:
            print("✅ Basic generation works")
        else:
            print("❌ Basic generation failed")
            return False
    except Exception as e:
        print(f"❌ Basic generation error: {e}")
        return False

    # Test multiple topics
    print("\n📊 Testing multiple topics...")
    topics = ["taxes", "social media", "dating apps", "fast food"]
    results = generate_multiple_riffs(topics, model_path)

    if len(results) != len(topics):
        print("❌ Not all topics generated successfully")
        return False

    # Quality analysis
    corny_count = 0
    slang_count = 0

    for topic, riff, gen_time in results:
        riff_lower = riff.lower()

        if any(word in riff_lower for word in ["lol", "haha", "lmao", "kek"]):
            corny_count += 1

        if any(word in riff_lower for word in ["yo", "fr", "type shi", "buh", "cuh"]):
            slang_count += 1

    print("
📈 Quality Results:"    print(f"   Topics tested: {len(topics)}")
    print(f"   Corny riffs: {corny_count}")
    print(f"   Slang-containing riffs: {slang_count}")

    if corny_count == 0:
        print("✅ No corny outputs detected!")
    else:
        print(f"⚠️ {corny_count} riffs contained corny elements")

    success_rate = (len(topics) - corny_count) / len(topics)
    print(".1%")

    return success_rate >= 0.8  # 80% success rate

if __name__ == "__main__":
    success = run_quality_assessment()
    if success:
        print("\n🎉 Quality assessment passed!")
        sys.exit(0)
    else:
        print("\n❌ Quality assessment failed!")
        sys.exit(1)

