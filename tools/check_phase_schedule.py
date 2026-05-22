import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.phase import PhaseConfig, resolve_phase


def main(opt):
    config = PhaseConfig(
        enabled=True,
        phase1_epochs=opt.phase1_epochs,
        phase2_epochs=opt.phase2_epochs,
        phase3_epochs=opt.phase3_epochs,
        phase2_img=tuple(opt.phase2_img) if opt.phase2_img else None,
        phase3_img=tuple(opt.phase3_img) if opt.phase3_img else None,
    )

    check_epochs = opt.epochs or sorted({
        0,
        max(config.phase1_epochs - 1, 0),
        config.phase1_epochs,
        max(config.phase3_start - 1, 0),
        config.phase3_start,
        29, 30, 289, 290, 359, 360,
    })
    for epoch in check_epochs:
        state = resolve_phase(epoch, config)
        print(f'{epoch},{state.name},rebuild={state.rebuild},img={state.img},rect={state.rect},mosaic={state.mosaic}')

    expected = {
        0: 'phase1',
        max(config.phase1_epochs - 1, 0): 'phase1',
        config.phase1_epochs: 'phase2',
        max(config.phase3_start - 1, 0): 'phase2',
        config.phase3_start: 'phase3',
    }
    for epoch, phase in expected.items():
        actual = resolve_phase(epoch, config).name
        if actual != phase:
            raise SystemExit(f'phase mismatch at epoch {epoch}: expected {phase}, got {actual}')

    print('phase schedule ok')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase1-epochs', type=int, default=290)
    parser.add_argument('--phase2-epochs', type=int, default=70)
    parser.add_argument('--phase3-epochs', type=int, default=40)
    parser.add_argument('--phase2-img', nargs=2, type=int, default=None)
    parser.add_argument('--phase3-img', nargs=2, type=int, default=None)
    parser.add_argument('--epochs', nargs='*', type=int, default=None)
    main(parser.parse_args())
