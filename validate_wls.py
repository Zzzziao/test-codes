"""Run reference agreement, backward, and finite-difference checks before SID.py."""
import ast
import json
from pathlib import Path

import cv2
import numpy as np
import scipy.sparse as sparse
import scipy.sparse.linalg as sl
import torch
import torch.nn.functional as F
from PIL import Image

from utils.WLS_utils import (differentiable_wls_filter, inverse_softplus,
                             dark_channel, estimate_atmospheric_light,
                             initial_transmission, atmospheric_scattering_model)


def reference_filter():
    # Execute the actual two reference functions unchanged, without importing
    # unrelated legacy CuPy code (CuPy is not needed by the SciPy reference).
    source = ast.parse(Path('utils/dehaze.py').read_text(encoding='utf-8'))
    functions = [node for node in source.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('process_difference_operator', 'wls_filter')]
    namespace = dict(np=np, cv2=cv2, sparse=sparse, sl=sl)
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'utils/dehaze.py', 'exec'), namespace)
    return namespace['wls_filter']


def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    image = Image.open('hazy input/1449.jpg').convert('RGB').resize((23, 17))
    hazy = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)[None].double() / 255
    airlight = estimate_atmospheric_light(hazy, dark_channel(hazy))
    transmission = initial_transmission(hazy, airlight)
    reference = reference_filter()
    results = {}
    cases = {'image_transmission': transmission,
             'random': 0.1 + 0.8 * torch.rand(1, 1, 17, 23, dtype=torch.double),
             'constant': torch.full((1, 1, 7, 11), 0.4, dtype=torch.double)}
    for name, value in cases.items():
        expected = reference(value[0, 0].numpy(), lambda_=0.35, alpha=4.0)
        actual = differentiable_wls_filter(value, 0.35, 4.0)
        error = np.abs(actual[0, 0].detach().numpy() - expected)
        results[name] = dict(max_error=float(error.max()), mean_error=float(error.mean()))
        np.testing.assert_allclose(actual[0, 0].numpy(), expected, atol=2e-5, rtol=2e-5)
        print(name, results[name])
    alpha_raw = torch.nn.Parameter(torch.tensor(inverse_softplus(3.0), dtype=torch.double))
    lambda_raw = torch.nn.Parameter(torch.tensor(inverse_softplus(0.35), dtype=torch.double))
    network = torch.nn.Sequential(torch.nn.Conv2d(3, 3, 3, padding=1), torch.nn.Sigmoid()).double()
    clear = network(hazy)

    def objective(a, l):
        t = differentiable_wls_filter(transmission, F.softplus(l), 1 + F.softplus(a))
        return (atmospheric_scattering_model(clear, t, airlight) - hazy).abs().mean()

    loss = objective(alpha_raw, lambda_raw)
    loss.backward()
    print(alpha_raw.grad, lambda_raw.grad)
    for name, parameter in [('alpha_raw', alpha_raw), ('lambda_raw', lambda_raw)]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad) and parameter.grad.abs() > 1e-12
        delta = 1e-4
        with torch.no_grad():
            args_plus = (alpha_raw + delta, lambda_raw) if name == 'alpha_raw' else (alpha_raw, lambda_raw + delta)
            args_minus = (alpha_raw - delta, lambda_raw) if name == 'alpha_raw' else (alpha_raw, lambda_raw - delta)
            finite_difference = (objective(*args_plus) - objective(*args_minus)) / (2 * delta)
        torch.testing.assert_close(parameter.grad, finite_difference, atol=2e-8, rtol=2e-3)
        results[name] = dict(gradient=parameter.grad.item(), finite_difference=finite_difference.item())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in network.parameters())
    # Also check dependence on the transmission/guidance, and independent batch solves.
    sample = (0.2 + 0.6 * torch.rand(2, 1, 3, 4, dtype=torch.double)).requires_grad_()
    assert torch.autograd.gradcheck(lambda x: differentiable_wls_filter(x, 0.35, 4.0), (sample,), atol=1e-5)
    for shape in [(1, 1, 1, 1), (1, 1, 1, 7), (1, 1, 7, 1)]:
        value = torch.full(shape, 0.4, dtype=torch.double, requires_grad=True)
        output = differentiable_wls_filter(value, 0.35, 4.0)
        torch.testing.assert_close(output, value)
        output.sum().backward()
        assert torch.isfinite(value.grad).all()
    from models import get_net
    net = get_net(3, 'skip', 'reflection', upsample_mode='bilinear')
    output = net(torch.rand(1, 3, 96, 96))[..., :65, :79]
    assert output.shape == (1, 3, 65, 79)
    results['status'] = 'All reference, gradient, batch, degenerate-size, and network-shape checks passed'
    Path('results').mkdir(exist_ok=True)
    Path('results/wls_validation.json').write_text(json.dumps(results, indent=2))
    print(results['status'])


if __name__ == '__main__':
    main()
