import torch

from bci.models.eegnet import EEGNet


def test_output_is_one_score_per_class():
    model = EEGNet()
    assert model(torch.randn(32, 22, 500)).shape == (32, 4)
    assert model(torch.randn(1, 22, 500)).shape == (1, 4)    # a single trial works too


def test_shapes_through_the_network():
    # Spatial filtering collapses the 22 electrodes to 1; pooling shrinks time
    # 500 -> 125 -> 15; 16 maps x 15 time steps feed the classifier.
    model = EEGNet()
    assert model.features(torch.randn(2, 1, 22, 500)).shape == (2, 16 * 15)


def test_parameter_count_is_small():
    # EEGNet-8,2 on 22 x 500 input: ~2.4k parameters (vs ~1M for a dense layer
    # on the flattened 11,000 inputs).
    assert sum(p.numel() for p in EEGNet().parameters()) == 2420


def test_every_parameter_gets_a_gradient():
    model = EEGNet()
    loss = torch.nn.functional.cross_entropy(model(torch.randn(8, 22, 500)), torch.arange(8) % 4)
    loss.backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
