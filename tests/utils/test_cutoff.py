from __future__ import annotations

import torch
from matgl.layers._basis import SphericalBesselFunction
from matgl.utils.cutoff import cosine_cutoff, polynomial_cutoff
from torch.testing import assert_close


def test_cosine():
    r = torch.linspace(1.0, 5.0, 11)
    three_body_cutoff = 4.0
    three_cutoff = cosine_cutoff(r, three_body_cutoff)
    assert_close(
        three_cutoff,
        torch.tensor([0.8536, 0.7270, 0.5782, 0.4218, 0.2730, 0.1464, 0.0545, 0.0062, 0.0000, 0.0000, 0.0000]),
        atol=1e-4,
        rtol=0.0,
    )


def test_polymonial_cutoff():
    # test fixed values
    r = torch.linspace(1.0, 5.0, 11)
    three_body_cutoff = 4.0
    envelope = polynomial_cutoff(r, three_body_cutoff)
    assert_close(
        envelope,
        torch.tensor([0.8965, 0.7648, 0.5931, 0.4069, 0.2352, 0.1035, 0.0266, 0.0012, 0.0000, 0.0000, 0.0000]),
        atol=1e-4,
        rtol=0.0,
    )

    # test behaviour smoothing a SBF with cutoff
    sbf = SphericalBesselFunction(max_l=1, max_n=5, cutoff=5)

    r = torch.linspace(1, 5, 10, requires_grad=True)
    envelope = polynomial_cutoff(r, cutoff=5, exponent=4)

    sbf_res = sbf(r)

    envelope_res = envelope[:, None] * sbf_res

    # assert that it is zero at the cutoff
    assert_close(sbf_res[-1, :], torch.zeros_like(sbf_res[-1, :]))
    assert_close(envelope_res[-1, :], torch.zeros_like(envelope_res[-1, :]))

    # assert derivatives vanish smoothly at the cutoff
    envelope_res.backward(torch.ones_like(envelope_res), retain_graph=True)
    assert r.grad[-1] == 0.0
    envelope_res.backward(torch.ones_like(envelope_res))
    assert r.grad[-1] == 0.0
