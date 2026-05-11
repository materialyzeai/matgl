"""Application-level wrappers built on top of ``matgl.models``.

Each submodule turns a bare graph network (M3GNet, CHGNet, TensorNet, ...)
into an end-user object that produces a physical quantity directly. The only
app currently shipped is :mod:`matgl.apps.pes`, which exposes
:class:`~matgl.apps.pes.Potential` -- a wrapper that derives forces (via
``-dE/dr``), stress (via ``V^{-1} dE/de``), and optionally the Hessian from
an energy-predicting graph model using PyTorch autograd.

Layers and backbones for building those underlying energy models live in
:mod:`matgl.layers` and :mod:`matgl.models`; the ``apps`` layer is concerned
with composition into a usable physical object, not with new architectures.
"""
