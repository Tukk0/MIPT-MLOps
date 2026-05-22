"""Download MNIST data to a local DVC-managed directory."""

from __future__ import annotations

from torchvision.datasets import MNIST


def download_mnist(target_dir: str = "./data", num_images: int | None = None) -> None:
    """Download MNIST dataset to *target_dir*."""
    MNIST(root=target_dir, train=True, download=True)
    MNIST(root=target_dir, train=False, download=True)
    print(f"MNIST downloaded to {target_dir}")
