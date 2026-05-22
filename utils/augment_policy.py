from dataclasses import dataclass


@dataclass
class AugmentPolicy:
    profile: str
    phase: str
    spider_web_p: float = 0.0
    gray_p: float = 0.0
    clahe_p: float = 0.0
    blur_p: float = 0.0
    compression_p: float = 0.0
    overexposure_p: float = 0.0
    patch_paste_p: float = 0.0
    hard_negative_p: float = 0.0

    @property
    def enabled(self):
        return self.profile != 'off'

    @classmethod
    def from_config(cls, profile='off', phase='baseline', hyp=None):
        hyp = hyp or {}
        profile = 'off' if profile in (None, '', 'none') else profile
        policy = cls(profile=profile, phase=phase or hyp.get('phase', 'baseline'))
        if profile == 'off':
            return policy

        policy.spider_web_p = float(hyp.get('cctv_spider_web_p', 0.05))
        policy.gray_p = float(hyp.get('cctv_gray_p', 0.15))
        policy.clahe_p = float(hyp.get('cctv_clahe_p', 0.20))
        policy.blur_p = float(hyp.get('cctv_blur_p', 0.15))
        policy.compression_p = float(hyp.get('cctv_compression_p', 0.25))
        policy.overexposure_p = float(hyp.get('cctv_overexposure_p', 0.10))

        if profile == 'cctv_paste' and policy.phase != 'phase3':
            policy.patch_paste_p = float(hyp.get('patch_paste_p', 0.10))
            policy.hard_negative_p = float(hyp.get('hard_negative_p', 0.10))
        return policy


def ensure_aug_option_defaults(opt):
    if not hasattr(opt, 'aug_profile'):
        opt.aug_profile = 'off'
    if not hasattr(opt, 'sampler_mode'):
        opt.sampler_mode = 'off'
    if opt.sampler_mode == 'none':
        opt.sampler_mode = 'off'
    if not hasattr(opt, 'aug_debug_samples'):
        opt.aug_debug_samples = 0
    if not hasattr(opt, 'hard_negative_manifest'):
        opt.hard_negative_manifest = ''
    return opt


def validate_aug_options(opt, parser=None):
    ensure_aug_option_defaults(opt)
    message = None
    if opt.sampler_mode == 'weighted' and getattr(opt, 'image_weights', False):
        message = '--sampler-mode weighted cannot be used with --image-weights'
    if opt.sampler_mode == 'weighted' and int(getattr(opt, 'world_size', 1)) > 1:
        message = '--sampler-mode weighted is single-GPU only until a distributed-aware sampler is implemented'
    if message:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return opt
