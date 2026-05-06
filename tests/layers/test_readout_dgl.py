from __future__ import annotations

import pytest
import torch
from torch import nn

import matgl

if matgl.config.BACKEND != "DGL":
    pytest.skip("Skipping DGL tests", allow_module_level=True)

from matgl.layers import BondExpansion, EmbeddingBlock
from matgl.layers._readout_dgl import (
    AttentiveFPReadout,
    GlobalPool,
    ReduceReadOut,
    Set2SetReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
    WeightedReadOutPair,
)


def _prepare_embedded_graph(
    g,
    *,
    dim_node_embedding: int = 16,
    dim_edge_embedding: int = 16,
    dim_state_feats: int = 16,
):
    """Prepare node_feat and edge_feat for DGL readout numerical tests."""

    bond_expansion = BondExpansion(
        rbf_type="SphericalBessel",
        max_n=3,
        max_l=3,
        cutoff=4.0,
        smooth=False,
    )
    bond_basis = bond_expansion(g.edata["bond_dist"])

    embed = EmbeddingBlock(
        degree_rbf=9,
        dim_node_embedding=dim_node_embedding,
        dim_edge_embedding=dim_edge_embedding,
        dim_state_feats=dim_state_feats,
        activation=nn.SiLU(),
        ntypes_node=2,
    )

    state_attr = torch.tensor(
        [1.0, 2.0],
        dtype=matgl.float_th,
        device=g.ndata["node_type"].device,
    )

    node_feat, edge_feat, state_feat = embed(
        g.ndata["node_type"],
        bond_basis,
        state_attr,
    )

    g.ndata["node_feat"] = node_feat
    g.edata["edge_feat"] = edge_feat

    return g, node_feat, edge_feat, state_feat


def _assert_close_to_expected(output: torch.Tensor, expected_values, *, rtol=1e-5, atol=1e-6):
    """Shared helper for numerical checks."""

    expected = torch.tensor(
        expected_values,
        dtype=output.dtype,
        device=output.device,
    )

    assert output.shape == expected.shape
    assert torch.isfinite(output).all()
    assert torch.allclose(output, expected, rtol=rtol, atol=atol)


def test_weighted_readout_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, _, _, _ = _prepare_embedded_graph(g1)

    read_out = WeightedReadOut(
        in_feats=16,
        dims=[32, 32],
        num_targets=4,
    )

    atomic_properties = read_out(g1)

    expected_values = [
        [
            0.028313392773270607,
            -0.06528592109680176,
            0.06681523472070694,
            -0.017671654000878334,
        ],
        [
            0.02230251207947731,
            -0.059122782200574875,
            0.07870473712682724,
            -0.010173093527555466,
        ],
    ]

    _assert_close_to_expected(atomic_properties, expected_values)


def test_weighted_atom_readout_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, _, _, _ = _prepare_embedded_graph(g1)

    read_out = WeightedAtomReadOut(
        in_feats=16,
        dims=[32, 32],
        activation=nn.SiLU(),
    )

    graph_properties = read_out(g1)

    expected_values = [
        [
            -0.014197269454598427,
            -0.06609135121107101,
            -0.046809203922748566,
            0.053692497313022614,
            0.013831745833158493,
            0.05812212824821472,
            -0.09599569439888,
            0.26410892605781555,
            -0.0010072104632854462,
            0.13218852877616882,
            0.08988366276025772,
            0.02111031673848629,
            -0.15803532302379608,
            0.046771831810474396,
            0.12612284719944,
            0.08061754703521729,
            -0.03074725717306137,
            0.04553832486271858,
            -0.04778451845049858,
            -0.13731348514556885,
            0.032929927110672,
            -0.05898449942469597,
            -0.051344603300094604,
            -0.049294304102659225,
            0.20951144397258759,
            0.07019517570734024,
            0.05725938081741333,
            0.08746682107448578,
            -0.12299986183643341,
            -0.08920607715845108,
            0.051440056413412094,
            0.19969353079795837,
        ]
    ]

    _assert_close_to_expected(graph_properties, expected_values)


def test_weighted_readout_pair_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, _, _, _ = _prepare_embedded_graph(g1)

    read_out = WeightedReadOutPair(
        in_feats=16,
        dims=[32, 32],
        num_targets=1,
    )

    pair_properties = read_out(g1)

    expected_values = [
        [
            [0.051521606743335724],
            [0.06976334005594254],
        ],
        [
            [0.06976334005594254],
            [0.06907030194997787],
        ],
    ]

    _assert_close_to_expected(pair_properties, expected_values)

    # Pair readout should be symmetric for this implementation.
    assert torch.allclose(
        pair_properties,
        pair_properties.transpose(0, 1),
        rtol=1e-5,
        atol=1e-6,
    )


def test_reduce_readout_node_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, node_feat, _, _ = _prepare_embedded_graph(g1)

    read_out = ReduceReadOut(
        op="mean",
        field="node_feat",
    )

    output = read_out(g1)
    expected = node_feat.mean(dim=0, keepdim=True)

    assert output.shape == (1, 16)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, expected, rtol=1e-6, atol=1e-7)


def test_reduce_readout_edge_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, _, edge_feat, _ = _prepare_embedded_graph(g1)

    read_out = ReduceReadOut(
        op="mean",
        field="edge_feat",
    )

    output = read_out(g1)
    expected = edge_feat.mean(dim=0, keepdim=True)

    assert output.shape == (1, 16)
    assert torch.isfinite(output).all()
    assert torch.allclose(output, expected, rtol=1e-6, atol=1e-7)


def test_set2set_node_readout_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, _, _, _ = _prepare_embedded_graph(
        g1,
        dim_node_embedding=16,
        dim_edge_embedding=32,
        dim_state_feats=16,
    )

    read_out = Set2SetReadOut(
        in_feats=16,
        n_iters=3,
        n_layers=3,
        field="node_feat",
    )

    output = read_out(g1)

    expected_values = [
        [
            -0.016146618872880936,
            -0.1164260283112526,
            0.05114629119634628,
            0.07186024636030197,
            -0.03709813207387924,
            -0.13907532393932343,
            -0.042047109454870224,
            -0.10545036196708679,
            0.030245458707213402,
            0.17485426366329193,
            -0.04700275883078575,
            0.03596976026892662,
            -0.13622251152992249,
            0.06921997666358948,
            -0.015076120384037495,
            0.17385578155517578,
            1.8063740730285645,
            0.7897505760192871,
            0.30854833126068115,
            -1.0275444984436035,
            0.0699705183506012,
            -0.2549362778663635,
            0.31435126066207886,
            -0.21318942308425903,
            0.10820049047470093,
            1.499506950378418,
            0.032316938042640686,
            -0.24378597736358643,
            -0.517692506313324,
            -0.3047972321510315,
            -0.5497528314590454,
            0.803704559803009,
        ]
    ]

    _assert_close_to_expected(output, expected_values)


def test_set2set_edge_readout_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, _, _, _ = _prepare_embedded_graph(
        g1,
        dim_node_embedding=16,
        dim_edge_embedding=32,
        dim_state_feats=16,
    )

    read_out = Set2SetReadOut(
        in_feats=32,
        n_iters=3,
        n_layers=3,
        field="edge_feat",
    )

    output = read_out(g1)

    expected_values = [
        [
            -0.07851210981607437,
            0.009802431799471378,
            0.0890706479549408,
            0.09195209294557571,
            0.018093086779117584,
            0.0025900532491505146,
            -0.055695630609989166,
            0.004012319725006819,
            -0.14321276545524597,
            -0.03777797147631645,
            0.0004021752974949777,
            0.03325974941253662,
            0.08786409348249435,
            0.03699878230690956,
            0.09728164225816727,
            -0.022624190896749496,
            -0.04277174547314644,
            0.041649557650089264,
            -0.07327521592378616,
            -0.00771187711507082,
            0.042439062148332596,
            0.06840945035219193,
            -0.02633018232882023,
            -0.06007075309753418,
            -0.020892338827252388,
            -0.08225294947624207,
            0.09764858335256577,
            -0.06266870349645615,
            -0.011683246120810509,
            -0.021052375435829163,
            -0.08837483078241348,
            -0.06698866933584213,
            -0.07774419337511063,
            -0.01976141147315502,
            0.15884588658809662,
            0.055197786539793015,
            -0.10971906781196594,
            -0.11007054895162582,
            -0.11388421058654785,
            -0.15689562261104584,
            -0.10687222331762314,
            -0.049844857305288315,
            0.14096996188163757,
            0.01977146789431572,
            -0.08038859069347382,
            -0.11734021455049515,
            0.10130810737609863,
            -0.1088956743478775,
            -0.001980651170015335,
            0.07319886982440948,
            -0.058681368827819824,
            -0.021106678992509842,
            -0.01237502321600914,
            -0.03825640305876732,
            -0.08217556774616241,
            -0.13315916061401367,
            -0.12182269245386124,
            -0.05015261098742485,
            0.1508958786725998,
            -0.06012832373380661,
            -0.125129833817482,
            -0.0024527222849428654,
            0.07324545085430145,
            0.18039757013320923,
        ]
    ]

    _assert_close_to_expected(output, expected_values)


def test_global_pool_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, node_feat, _, _ = _prepare_embedded_graph(g1)

    g_attr = torch.zeros(
        (1, 16),
        dtype=matgl.float_th,
        device=node_feat.device,
    )

    pool = GlobalPool(feat_size=16)

    g_feat, attention_weights = pool(
        g1,
        node_feat,
        g_attr,
        get_node_weight=True,
    )

    expected_g_feat_values = [
        [
            -0.09950394928455353,
            0.21906785666942596,
            0.24901634454727173,
            0.033706746995449066,
            -0.013436689041554928,
            -0.2596094608306885,
            0.04884521663188934,
            -0.004793053492903709,
            0.10356846451759338,
            0.16007888317108154,
            0.1606583595275879,
            0.001753549207933247,
            0.03743843734264374,
            0.16566577553749084,
            0.025987280532717705,
            -0.16207806766033173,
        ]
    ]

    expected_attention_values = [
        [0.5446093678474426],
        [0.45539069175720215],
    ]

    _assert_close_to_expected(g_feat, expected_g_feat_values)
    _assert_close_to_expected(attention_weights, expected_attention_values)

    assert torch.allclose(
        attention_weights.sum(),
        torch.tensor(
            1.0,
            dtype=attention_weights.dtype,
            device=attention_weights.device,
        ),
        rtol=1e-6,
        atol=1e-7,
    )

    g_feat_without_weights = pool(g1, node_feat, g_attr)

    assert g_feat_without_weights.shape == (1, 16)
    assert torch.isfinite(g_feat_without_weights).all()
    assert torch.allclose(g_feat, g_feat_without_weights, rtol=1e-6, atol=1e-7)


def test_attentive_fp_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, node_feat, _, _ = _prepare_embedded_graph(g1)

    pool = AttentiveFPReadout(
        feat_size=16,
        num_timesteps=2,
    )

    g_feat, attention_weights = pool(
        g1,
        node_feat,
        get_node_weight=True,
    )

    expected_g_feat_values = [
        [
            1.6055431365966797,
            0.9094326496124268,
            -0.11220824718475342,
            -0.2558125853538513,
            0.11083554476499557,
            -0.3912426233291626,
            0.606181263923645,
            -0.00851428508758545,
            0.3602955937385559,
            0.9617499113082886,
            -0.059919342398643494,
            -0.231099933385849,
            0.08161452412605286,
            0.10026505589485168,
            -0.20568835735321045,
            0.03129486367106438,
        ]
    ]

    expected_attention_values = [
        [0.5479905605316162, 0.5599585175514221],
        [0.4520094692707062, 0.4400414824485779],
    ]

    _assert_close_to_expected(g_feat, expected_g_feat_values)
    _assert_close_to_expected(attention_weights, expected_attention_values)

    assert torch.allclose(
        attention_weights.sum(dim=0),
        torch.ones(
            attention_weights.shape[1],
            dtype=attention_weights.dtype,
            device=attention_weights.device,
        ),
        rtol=1e-6,
        atol=1e-7,
    )

    g_feat_without_weights = pool(g1, node_feat)

    assert g_feat_without_weights.shape == (1, 16)
    assert torch.isfinite(g_feat_without_weights).all()
    assert torch.allclose(g_feat, g_feat_without_weights, rtol=1e-6, atol=1e-7)
