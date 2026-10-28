from __future__ import annotations

import pytest
import torch
from torch import nn
from torch_geometric.data import Data

import matgl
from matgl.layers import BondExpansion, EmbeddingBlock
from matgl.layers._readout import (
    EdgeSet2Set,
    ReduceReadOut,
    Set2SetReadOut,
    WeightedAtomReadOut,
    WeightedReadOut,
)


def _prepare_embedded_graph(
    g,
    *,
    dim_node_embedding: int = 16,
    dim_edge_embedding: int = 16,
    dim_state_feats: int = 16,
):
    """Prepare node_feat and edge_feat for PYG readout numerical tests."""
    bond_expansion = BondExpansion(
        rbf_type="SphericalBessel",
        max_n=3,
        max_l=3,
        cutoff=4.0,
        smooth=False,
    )
    bond_basis = bond_expansion(g.bond_dist)

    embed = EmbeddingBlock(
        degree_rbf=9,
        dim_node_embedding=dim_node_embedding,
        dim_edge_embedding=dim_edge_embedding,
        dim_state_feats=dim_state_feats,
        activation=nn.SiLU(),
        ntypes_node=2,
    )

    state_attr = torch.tensor([1.0, 2.0], dtype=matgl.float_th)

    node_feat, edge_feat, state_feat = embed(g.node_type, bond_basis, state_attr)

    g.node_feat = node_feat
    g.edge_feat = edge_feat

    return g, node_feat, edge_feat, state_feat


def _assert_close_to_expected(output: torch.Tensor, expected_values, *, rtol=1e-5, atol=1e-6):
    """Shared helper for numerical checks."""
    expected = torch.tensor(expected_values, dtype=output.dtype, device=output.device)
    assert output.shape == expected.shape
    assert torch.isfinite(output).all()
    assert torch.allclose(output, expected, rtol=rtol, atol=atol)


def test_weighted_readout_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, _, _, _ = _prepare_embedded_graph(g1)

    read_out = WeightedReadOut(in_feats=16, dims=[32, 32], num_targets=4)
    atomic_properties = read_out(g1.node_feat)

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

    read_out = WeightedAtomReadOut(in_feats=16, dims=[32, 32], activation=nn.SiLU())
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


def test_reduce_readout_node_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, node_feat, _, _ = _prepare_embedded_graph(g1)

    read_out = ReduceReadOut(op="mean", field="node_feat")
    output = read_out(g1)
    expected = node_feat.mean(dim=0, keepdim=True)

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

    read_out = Set2SetReadOut(in_feats=16, n_iters=3, n_layers=3, field="node_feat")
    output = read_out(g1.node_feat, index=g1.batch)

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


def test_edge_set2set_numbers(graph_MoS):
    torch.manual_seed(42)

    _, g1, _ = graph_MoS
    g1, _, _, _ = _prepare_embedded_graph(
        g1,
        dim_node_embedding=16,
        dim_edge_embedding=32,
        dim_state_feats=16,
    )

    read_out = EdgeSet2Set(input_dim=32, n_iters=3, n_layers=3)
    edge_batch = torch.zeros(g1.num_edges, dtype=torch.long)
    output = read_out(g1.edge_feat, edge_batch)

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


# ---------------------------------------------------------------------------
# ReduceReadOut
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "max"])
def test_reduce_readout_unbatched(op):
    """The non-batched path collapses (N, F) -> (1, F) for every supported op."""
    x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = ReduceReadOut(op=op).forward(x)
    assert out.shape == (1, 3)
    assert torch.isfinite(out).all()
    if op == "sum":
        assert torch.allclose(out, torch.tensor([[5.0, 7.0, 9.0]]))
    elif op == "mean":
        assert torch.allclose(out, torch.tensor([[2.5, 3.5, 4.5]]))
    else:  # max
        assert torch.allclose(out, torch.tensor([[4.0, 5.0, 6.0]]))


@pytest.mark.parametrize("op", ["sum", "mean", "max"])
def test_reduce_readout_batched(op):
    """The batched path scatters into per-graph rows using the `batch` index."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    batch = torch.tensor([0, 0, 1, 1])
    out = ReduceReadOut(op=op).forward(x, batch=batch)
    assert out.shape == (2, 2)
    assert torch.isfinite(out).all()
    if op == "sum":
        assert torch.allclose(out, torch.tensor([[4.0, 6.0], [12.0, 14.0]]))
    elif op == "mean":
        assert torch.allclose(out, torch.tensor([[2.0, 3.0], [6.0, 7.0]]))
    else:  # max
        assert torch.allclose(out, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))


def test_reduce_readout_invalid_op_raises():
    with pytest.raises(ValueError, match="op must be"):
        ReduceReadOut(op="bogus")


# ---------------------------------------------------------------------------
# WeightedReadOut
# ---------------------------------------------------------------------------


def test_weighted_readout_tensor_numbers():
    """WeightedReadOut maps per-node features to (N, num_targets) with fixed seed."""
    torch.manual_seed(42)
    x = torch.randn(5, 4)
    wr = WeightedReadOut(in_feats=4, dims=[8, 8], num_targets=3)
    out = wr(x)

    expected_values = [
        [-0.07476293295621872, -0.13569995760917664, -0.0838092565536499],
        [-0.0751706138253212, -0.13388681411743164, -0.0648883655667305],
        [-0.07388679683208466, -0.11860352009534836, -0.06600876152515411],
        [-0.0732363611459732, -0.12554557621479034, -0.06692913919687271],
        [-0.06959028542041779, -0.12596376240253448, -0.07409191131591797],
    ]

    _assert_close_to_expected(out, expected_values)


# ---------------------------------------------------------------------------
# WeightedAtomReadOut
# ---------------------------------------------------------------------------


def test_weighted_atom_readout_unbatched_numbers():
    """Unbatched WeightedAtomReadOut collapses all nodes to a single (1, dim) row."""
    torch.manual_seed(42)
    x = torch.randn(6, 4)
    war = WeightedAtomReadOut(in_feats=4, dims=[8, 8], activation=nn.SiLU())
    out = war(x)

    expected_values = [
        [
            0.019682496786117554,
            0.012260183691978455,
            0.1443532407283783,
            0.2507040798664093,
            0.1520005464553833,
            -0.14217662811279297,
            0.04660123586654663,
            -0.04173845052719116,
        ]
    ]

    _assert_close_to_expected(out, expected_values)


def test_weighted_atom_readout_batched_numbers():
    """Batched WeightedAtomReadOut produces one row per graph, graphs produce different outputs."""
    torch.manual_seed(42)
    x = torch.randn(6, 4)
    batch = torch.tensor([0, 0, 0, 1, 1, 1])
    war = WeightedAtomReadOut(in_feats=4, dims=[8, 8], activation=nn.SiLU())
    out = war(x, batch=batch)

    expected_values = [
        [
            0.08378858864307404,
            0.030509421601891518,
            0.13405263423919678,
            0.24054312705993652,
            0.11397531628608704,
            -0.16801601648330688,
            0.02209596149623394,
            0.0015756934881210327,
        ],
        [
            -0.05510534346103668,
            -0.00902985967695713,
            0.15637019276618958,
            0.2625580430030823,
            0.1963617205619812,
            -0.11203167587518692,
            0.07518972456455231,
            -0.09226985275745392,
        ],
    ]

    _assert_close_to_expected(out, expected_values)
    assert not torch.allclose(out[0], out[1]), "two distinct graphs must produce different readouts"


# ---------------------------------------------------------------------------
# Explicit num_graphs (trailing graphs without nodes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["sum", "mean", "max"])
def test_reduce_readout_num_graphs_keeps_trailing_empty_graphs(op):
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    batch = torch.tensor([0, 0])
    out = ReduceReadOut(op=op)(x, batch, num_graphs=3)
    assert out.shape == (3, 2)
    assert torch.equal(out[1:], torch.zeros(2, 2))
    assert torch.allclose(out[:1], ReduceReadOut(op=op)(x, batch))


def test_reduce_readout_max_empty_middle_graph_is_zero():
    x = torch.tensor([[1.0], [5.0]])
    out = ReduceReadOut(op="max")(x, torch.tensor([0, 2]), num_graphs=3)
    assert torch.equal(out, torch.tensor([[1.0], [0.0], [5.0]]))


def test_weighted_atom_readout_num_graphs_keeps_trailing_empty_graphs():
    torch.manual_seed(0)
    x = torch.randn(4, 4)
    batch = torch.tensor([0, 0, 1, 1])
    war = WeightedAtomReadOut(in_feats=4, dims=[8, 8], activation=nn.SiLU())
    out = war(x, batch, num_graphs=3)
    assert out.shape == (3, 8)
    assert torch.allclose(out[:2], war(x, batch))
    assert torch.equal(out[2], torch.zeros(8))


def test_readouts_accept_legacy_graph_input():
    """Calling with a PyG graph carrying ``node_feat`` (the old ``matgl.layers`` API) still works."""
    torch.manual_seed(0)
    x = torch.randn(4, 4)
    batch = torch.tensor([0, 0, 1, 1])
    graph = Data(node_feat=x, batch=batch)
    war = WeightedAtomReadOut(in_feats=4, dims=[8, 8], activation=nn.SiLU())
    assert torch.allclose(war(graph), war(x, batch))
    assert torch.allclose(ReduceReadOut("sum")(graph), ReduceReadOut("sum")(x, batch))
