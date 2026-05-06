"""Module implementing various data transformers for PyTorch."""

from __future__ import annotations

import abc

import torch


class Transformer(metaclass=abc.ABCMeta):
    """Abstract base class defining a data transformer."""

    @abc.abstractmethod
    def transform(self, data: torch.Tensor):
        """Transformation to be performed on data.

        Args:
            data: Input data

        Returns:
            Transformed data.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse_transform(self, data: torch.Tensor):
        """Inverse transformation to be performed on data.

        Args:
            data: Input data

        Returns:
            Inverse-transformed data.
        """
        raise NotImplementedError


class Normalizer(Transformer):
    """Performs a scaling of the data by centering to the mean and dividing by the standard deviation."""

    def __init__(self, mean: float, std: float):
        """Initialize the Normalizer.

        Args:
            mean: Mean of the data
            std: Standard deviation of the data.
        """
        self.mean = mean
        self.std = std

    def transform(self, data):
        """z-score the data by subtracting the mean and dividing by the standard deviation.

        Args:
            data: Input data

        Returns:
            Scaled data
        """
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        """Invert the scaling.

        Args:
            data: Scaled data

        Returns:
            Unscaled data
        """
        return data * self.std + self.mean

    def __repr__(self):
        mean, std = self.mean, self.std
        return f"Normalizer({mean=}, {std=})"

    @classmethod
    def from_data(cls, data):
        """Create Normalizer from data.

        Args:
            data: Input data.

        Returns:
            Normalizer
        """
        data = torch.tensor(data)
        return Normalizer(torch.mean(data), torch.std(data))


class LogTransformer(Transformer):
    """Performs a natural log of the data."""

    def transform(self, data):
        """Take the log of the data.

        Args:
            data: Input data

        Returns:
            Scaled data
        """
        return torch.log(data)

    def inverse_transform(self, data):
        """Invert the log (exp).

        Args:
            data: Input data

        Returns:
            exp(data)
        """
        return torch.exp(data)

    def __repr__(self):
        return "LogTransformer"
