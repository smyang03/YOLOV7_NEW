from dataclasses import dataclass
from typing import Optional, Tuple


def _tuple_or_none(value):
    if value is None:
        return None
    return tuple(int(x) for x in value)


def _on_off_to_bool(value, default=True):
    if value is None:
        return default
    return str(value).lower() == 'on'


@dataclass(frozen=True)
class PhaseConfig:
    enabled: bool = False
    phase1_epochs: int = 290
    phase2_epochs: int = 70
    phase3_epochs: int = 40
    phase2_img: Optional[Tuple[int, int]] = None
    phase3_img: Optional[Tuple[int, int]] = None
    phase2_rect: bool = True
    phase2_mosaic: bool = True
    phase3_mosaic: bool = False

    @classmethod
    def from_opt(cls, opt):
        cfg_name = str(getattr(opt, 'cfg', '')).lower()
        rect_default = getattr(opt, 'rect_size_w6', None) if 'w6' in cfg_name else getattr(opt, 'rect_size_l', None)
        phase2_img = _tuple_or_none(getattr(opt, 'phase2_img', None)) or _tuple_or_none(rect_default)
        phase3_img = _tuple_or_none(getattr(opt, 'phase3_img', None)) or _tuple_or_none(rect_default)
        return cls(
            enabled=getattr(opt, 'phase_train', 'off') == 'on',
            phase1_epochs=int(getattr(opt, 'phase1_epochs', 290)),
            phase2_epochs=int(getattr(opt, 'phase2_epochs', 70)),
            phase3_epochs=int(getattr(opt, 'phase3_epochs', 40)),
            phase2_img=phase2_img,
            phase3_img=phase3_img,
            phase2_rect=bool(getattr(opt, 'phase2_rect', True)),
            phase2_mosaic=_on_off_to_bool(getattr(opt, 'phase2_mosaic', 'on'), True),
            phase3_mosaic=_on_off_to_bool(getattr(opt, 'phase3_mosaic', 'off'), False),
        )

    @property
    def phase2_start(self):
        return self.phase1_epochs

    @property
    def phase3_start(self):
        return self.phase1_epochs + self.phase2_epochs

    def boundaries(self):
        return {
            'phase1_end': self.phase1_epochs,
            'phase2_end': self.phase3_start,
            'phase3_end': self.phase1_epochs + self.phase2_epochs + self.phase3_epochs,
        }


@dataclass(frozen=True)
class PhaseState:
    name: str
    index: int
    epoch: int
    start_epoch: int
    end_epoch: Optional[int]
    img: Optional[Tuple[int, int]] = None
    rect: Optional[bool] = None
    mosaic: Optional[bool] = None
    rebuild: bool = False

    @property
    def train_imgsz(self):
        return max(self.img) if self.img else None


def resolve_phase(epoch, config):
    if not config.enabled:
        return PhaseState('baseline', 0, epoch, 0, None, rebuild=False)
    if epoch < config.phase1_epochs:
        return PhaseState('phase1', 1, epoch, 0, config.phase1_epochs, rebuild=False)
    if epoch < config.phase3_start:
        return PhaseState(
            'phase2', 2, epoch, config.phase2_start, config.phase3_start,
            img=config.phase2_img, rect=config.phase2_rect, mosaic=config.phase2_mosaic, rebuild=True)
    return PhaseState(
        'phase3', 3, epoch, config.phase3_start,
        config.phase1_epochs + config.phase2_epochs + config.phase3_epochs,
        img=config.phase3_img, rect=True, mosaic=config.phase3_mosaic, rebuild=True)


def phase_changed(previous, current):
    return previous is not None and previous.name != current.name
