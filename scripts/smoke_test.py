#!/usr/bin/env python3
"""Data-free structural smoke test for all implemented model variants."""
import torch

from sira.models import DeepJSCC, SIRA_METHODS


def count_parameters(model, trainable_only=False):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad or not trainable_only
    )


def main():
    torch.manual_seed(42)
    x = torch.rand(2, 3, 64, 64)
    methods = ('cnn', 'semantic') + SIRA_METHODS

    for method in methods:
        model = DeepJSCC(
            method=method,
            latent_ch=2,
            channel='awgn',
            input_size=64,
        ).eval()
        with torch.no_grad():
            output = model(x, snr_db=0.0)

        assert output.shape == x.shape, (method, output.shape)
        if method in SIRA_METHODS:
            assert not any(p.requires_grad for p in model.encoder.parameters())
            assert not any(p.requires_grad for p in model.decoder.parameters())
            assert model._last_sira['power_map'].shape == (2, 1, 32, 32)

        print(
            f'{method:>14}: output={tuple(output.shape)} '
            f'trainable={count_parameters(model, trainable_only=True):,} '
            f'total={count_parameters(model):,}'
        )

    no_r = DeepJSCC(method='sira_b2_no_r', latent_ch=2, input_size=64)
    assert not any(name.startswith('R.') for name, _ in no_r.named_parameters())
    print('Smoke test passed.')


if __name__ == '__main__':
    main()
