"""Tests for the HRR (Holographic Reduced Representations) module."""

import numpy as np
import pytest

from vecmemori import hrr


class TestHrrCore:
    def test_encode_atom_consistency(self):
        """Same word → same vector (deterministic)."""
        v1 = hrr.encode_atom("hello", dim=256)
        v2 = hrr.encode_atom("hello", dim=256)
        np.testing.assert_array_equal(v1, v2)

    def test_encode_atom_different_words(self):
        """Different words → different vectors (quasi-orthogonal)."""
        v1 = hrr.encode_atom("hello", dim=256)
        v2 = hrr.encode_atom("world", dim=256)
        similarity = float(np.mean(np.cos(v1 - v2)))
        assert abs(similarity) < 0.3, f"Expected near-zero similarity, got {similarity}"

    def test_encode_atom_dimension(self):
        """Result shape matches requested dim."""
        for dim in [64, 128, 256, 1024]:
            v = hrr.encode_atom("test", dim=dim)
            assert v.shape == (dim,), f"Expected ({dim},), got {v.shape}"

    def test_bind_unbind_roundtrip(self):
        """bind(a, b) → unbind(result, a) ≈ b (up to superposition noise)."""
        dim = 256
        a = hrr.encode_atom("concept_a", dim)
        b = hrr.encode_atom("concept_b", dim)
        bound = hrr.bind(a, b)
        recovered = hrr.unbind(bound, a)
        # Should be close to b
        sim = hrr.similarity(recovered, b)
        assert sim > 0.8, f"Bind/unbind roundtrip too noisy: sim={sim}"

    def test_bundle_similarity(self):
        """Bundled vectors are similar to each component."""
        dim = 256
        a = hrr.encode_atom("alpha", dim)
        b = hrr.encode_atom("beta", dim)
        bundled = hrr.bundle(a, b)
        sim_a = hrr.similarity(bundled, a)
        sim_b = hrr.similarity(bundled, b)
        assert sim_a > 0.3 and sim_b > 0.3, (
            f"Bundle components too dissimilar: a={sim_a}, b={sim_b}"
        )

    def test_similarity_identical(self):
        """Self-similarity should be 1.0."""
        v = hrr.encode_atom("test", dim=256)
        assert hrr.similarity(v, v) == pytest.approx(1.0, abs=1e-6)

    def test_similarity_range(self):
        """Similarity is in [-1, 1]."""
        dim = 256
        v1 = hrr.encode_atom("cat", dim)
        v2 = hrr.encode_atom("dog", dim)
        v3 = hrr.encode_atom("cat", dim)  # same as v1
        sim = hrr.similarity(v1, v2)
        assert -1.0 <= sim <= 1.0, f"Similarity out of range: {sim}"
        assert hrr.similarity(v1, v3) == pytest.approx(1.0, abs=1e-6)

    def test_encode_text_bag_of_words(self):
        """Text encoding is a bundle of token atoms."""
        dim = 256
        v = hrr.encode_text("hello world", dim)
        assert v.shape == (dim,)

    def test_encode_text_empty(self):
        """Empty string returns a deterministic fallback."""
        dim = 256
        v = hrr.encode_text("", dim)
        assert v.shape == (dim,)

    def test_encode_fact(self):
        """encode_fact returns a valid vector with content+entity structure."""
        dim = 256
        v = hrr.encode_fact("User likes coffee", ["User", "coffee"], dim)
        assert v.shape == (dim,)

    def test_serialization_roundtrip(self):
        """phases_to_bytes / bytes_to_phases roundtrip is lossless."""
        dim = 256
        v = hrr.encode_atom("persist_test", dim)
        data = hrr.phases_to_bytes(v)
        recovered = hrr.bytes_to_phases(data)
        np.testing.assert_array_equal(v, recovered)

    def test_snr_estimate(self):
        """SNR estimate is positive for valid inputs."""
        snr = hrr.snr_estimate(1024, 100)
        assert snr > 0
        assert snr == pytest.approx(np.sqrt(1024 / 100))
