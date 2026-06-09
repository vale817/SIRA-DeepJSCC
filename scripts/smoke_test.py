#!/usr/bin/env python3
"""Data-free structural smoke test for all implemented model variants."""
import torch

from sira.models import DeepJSCC, SIRA_METHODS, stabilized_water_filling_power


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
            assert model._last_sira['power_map_spatial'].shape == (2, 1, 32, 32)
            assert model._last_sira['power_symbol'].shape == (2, 2, 32, 32)
            assert model._last_sira['effective_power_symbol'].shape == (2, 2, 32, 32)
            assert model._last_sira['effective_power_map_spatial'].shape == (2, 1, 32, 32)
            assert model._last_sira['power_alpha'].shape == (2, 1, 1, 1)
            assert model._last_sira['transmit_energy'].shape == (2,)
            assert model._last_sira['low_snr_blend_lambda'].shape == (2, 1, 1, 1)
            power_sum = model._last_sira['power_symbol'].reshape(2, -1).sum(dim=1)
            assert torch.allclose(power_sum, torch.full_like(power_sum, 2 * 32 * 32), rtol=1e-4)
            assert torch.allclose(
                model._last_sira['transmit_energy'],
                torch.ones(2),
                rtol=1e-5,
                atol=1e-6,
            )
            assert model._last_sira['semantic_risk'].shape == (2, 2, 32, 32)

        print(
            f'{method:>14}: output={tuple(output.shape)} '
            f'trainable={count_parameters(model, trainable_only=True):,} '
            f'total={count_parameters(model):,}'
        )

    no_r = DeepJSCC(method='sira_b2_no_r', latent_ch=2, input_size=64)
    assert not any(name.startswith('R.') for name, _ in no_r.named_parameters())

    risk = torch.tensor([[[[0.01, 0.1], [1.0, 10.0]]]]).expand(2, -1, -1, -1)
    gamma = torch.tensor([10.0 ** (-2.0 / 10.0), 10.0 ** (15.0 / 10.0)])
    p_stable, _, _, blend = stabilized_water_filling_power(risk, gamma)
    assert blend[0].item() > blend[1].item()
    assert p_stable[0].min().item() > 0.0
    assert torch.allclose(
        p_stable.reshape(2, -1).sum(dim=1),
        torch.full((2,), 4.0),
        rtol=1e-5,
    )

    hard = DeepJSCC(
        method='sira_b2_init', latent_ch=2, input_size=64,
        allocation_mode='hard',
    ).eval()
    soft = DeepJSCC(
        method='sira_b2_init', latent_ch=2, input_size=64,
        allocation_mode='soft',
    ).eval()
    soft.load_state_dict(hard.state_dict())
    with torch.no_grad():
        hard(x, snr_db=-2.0)
        soft(x, snr_db=-2.0)
    assert hard._last_sira['allocation_mode'] == 'hard'
    assert soft._last_sira['allocation_mode'] == 'soft'
    assert torch.count_nonzero(hard._last_sira['low_snr_blend_lambda']) == 0
    assert torch.all(soft._last_sira['low_snr_blend_lambda'] > 0)
    assert not torch.allclose(
        hard._last_sira['power_prior'],
        soft._last_sira['power_prior'],
    )
    print('Smoke test passed.')


if __name__ == '__main__':
    main()
