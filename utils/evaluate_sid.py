"""Compare a saved SID pilot with its paired clear reference at the run resolution."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='results/sid_joint_128')
    parser.add_argument('--reference', default='hazy input/1449-gt.jpg')
    args = parser.parse_args()
    directory = Path(args.output_dir)
    hazy = Image.open(directory / 'hazy.png').convert('RGB')
    output = Image.open(directory / 'dehazed.png').convert('RGB')
    reference = Image.open(args.reference).convert('RGB').resize(hazy.size, Image.Resampling.LANCZOS)
    if output.size != hazy.size:
        raise ValueError('Hazy and dehazed dimensions must match')

    def psnr(left, right):
        error = np.mean(((np.asarray(left, dtype=float) - np.asarray(right, dtype=float)) / 255) ** 2)
        return float(-10 * np.log10(max(error, 1e-12)))

    history = json.loads((directory / 'history.json').read_text())
    metrics = dict(resolution=list(hazy.size), iterations=len(history),
                   hazy_reference_psnr=psnr(hazy, reference),
                   dehazed_reference_psnr=psnr(output, reference),
                   initial=history[0], final=history[-1])
    metrics['reference_psnr_gain'] = metrics['dehazed_reference_psnr'] - metrics['hazy_reference_psnr']
    comparison = Image.new('RGB', (hazy.width * 3, hazy.height))
    for index, image in enumerate((hazy, output, reference)):
        comparison.paste(image, (index * hazy.width, 0))
    comparison.save(directory / 'comparison_reference.png')
    (directory / 'evaluation.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()



gt = Image.open('hazy input/1449-gt.png').convert('RGB').resize((128, 128), Image.Resampling.LANCZOS)
gt.save('results/1449 2000 it/gt.png')

folder = Path(__file__).resolve().parent / "results" / "1449 2000 it"

reference_path = folder / "gt.png"
prediction_path = folder / "out_02000.png"

def load_rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float64)


reference = load_rgb(reference_path)
prediction = load_rgb(prediction_path)

if reference.shape != prediction.shape:
    raise ValueError(
        f"Size of images different：Ground truth {reference.shape}，Dehazed {prediction.shape}"
    )

mse = np.mean((reference - prediction) ** 2)
psnr = float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)

print(f"Reference:  {reference_path}")
print(f"Prediction: {prediction_path}")
print(f"RGB PSNR: {psnr:.4f} dB")