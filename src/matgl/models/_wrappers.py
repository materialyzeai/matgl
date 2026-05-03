"""Implementations of pseudo-models that wrap other models."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._core import MatGLModel

if TYPE_CHECKING:
    from torch import nn

    from matgl.data.transformer import Transformer


class TransformedTargetModel(MatGLModel):
    """Model that transforms the target prior to training and inverts the transform for predictions.

    Modelled after scikit-learn's TransformedTargetRegressor. This wrapper is almost never used for
    training (the general idea is to use the transformed target for loss computation); instead, it
    is created after a model has been fitted for serialization, so end users can call the model to
    perform predictions without having to worry about what target transformations have been
    performed.
    """

    # Model version number.
    __version__ = 1

    def __init__(self, model: nn.Module, target_transformer: Transformer):
        """Initialize the TransformedTargetModel.

        Args:
            model (nn.Module): Model to wrap.
            target_transformer (Transformer): Transformer to use for target transformation.
        """
        super().__init__()
        self.save_args(locals())
        self.model = model
        self.transformer = target_transformer

    def forward(self, *args, **kwargs):
        """Run the wrapped model and inverse-transform the output.

        Args:
            *args: Passthrough to parent model.forward method.
            **kwargs: Passthrough to parent model.forward method.

        Returns:
            Inverse transformed output.
        """
        output = self.model.forward(*args, **kwargs)
        return self.transformer.inverse_transform(output)

    def __repr__(self):
        return f"{type(self).__name__}:\n\tModel: {self.model!r}\n\tTransformer: {self.transformer!r}"

    def predict_structure(self, *args, **kwargs):
        """Pass through to parent model.predict_structure with inverse transform.

        Args:
            *args: Pass-through to self.model.predict_structure.
            **kwargs: Pass-through to self.model.predict_structure.

        Returns:
            Transformed answer.
        """
        return self.transformer.inverse_transform(self.model.predict_structure(*args, **kwargs))
