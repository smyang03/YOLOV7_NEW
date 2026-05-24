from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ensure_structure_option_defaults(opt):
    if not hasattr(opt, 'det_head'):
        opt.det_head = 'anchor'
    if not hasattr(opt, 'anchor_free_levels'):
        opt.anchor_free_levels = 'p3p4p5'
    if not hasattr(opt, 'lambda_free'):
        opt.lambda_free = 1.0
    if not hasattr(opt, 'fcos_center_radius'):
        opt.fcos_center_radius = 1.5
    if not hasattr(opt, 'fcos_score_mode'):
        opt.fcos_score_mode = 'sqrt_cls_centerness'
    if not hasattr(opt, 'fcos_loss_box'):
        opt.fcos_loss_box = 'giou'
    if not hasattr(opt, 'p2_head'):
        opt.p2_head = 'none'
    if not hasattr(opt, 'neck_mod'):
        opt.neck_mod = 'none'
    if not hasattr(opt, 'psa_level'):
        opt.psa_level = 'none'
    if not hasattr(opt, 'optional_decision'):
        opt.optional_decision = ''
    cfg_name = Path(str(getattr(opt, 'cfg', '') or '')).name.lower()
    if opt.det_head == 'anchor' and 'fcos' in cfg_name and 'decode' not in cfg_name:
        opt.det_head = 'hybrid' if 'hybrid' in cfg_name else 'fcos'
    if opt.anchor_free_levels == 'p3p4p5' and 'w6' in cfg_name and 'p2' in cfg_name and 'fcos' in cfg_name:
        opt.anchor_free_levels = 'p2'
    elif opt.anchor_free_levels == 'p3p4p5' and 'w6' in cfg_name and 'fcos' in cfg_name:
        opt.anchor_free_levels = 'p2p3p4p5p6'
    if opt.p2_head == 'none' and 'w6' in cfg_name and 'fcos' in cfg_name:
        opt.p2_head = 'fcos'
    elif opt.p2_head == 'none' and 'w6' in cfg_name and 'p2' in cfg_name:
        opt.p2_head = 'anchor'
    if opt.neck_mod == 'none' and 'w6' in cfg_name and 'scdown' in cfg_name:
        opt.neck_mod = 'scdown'
    elif opt.neck_mod == 'none' and 'w6' in cfg_name and 'psa' in cfg_name:
        opt.neck_mod = 'psa'
    elif opt.neck_mod == 'none' and 'w6' in cfg_name and 'gelan' in cfg_name:
        opt.neck_mod = 'gelan'
    if opt.neck_mod == 'psa' and opt.psa_level == 'none':
        opt.psa_level = 'p5'
    return opt


def _resolve_existing_path(path):
    if not path:
        return None
    p = Path(path)
    candidates = [p, ROOT / p]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _is_w6_cfg(opt):
    cfg_name = Path(str(getattr(opt, 'cfg', '') or '')).name.lower()
    return 'w6' in cfg_name


def active_optional_experiments(opt):
    ensure_structure_option_defaults(opt)
    active = []
    if getattr(opt, 'aux', 'auto') == 'on' and not _is_w6_cfg(opt):
        active.append('l_aux_on')
    if getattr(opt, 'p2_head', 'none') == 'fcos' or getattr(opt, 'det_head', 'anchor') in ('fcos', 'hybrid'):
        active.append('fcos_p2' if getattr(opt, 'anchor_free_levels', '') == 'p2' else 'fcos')
    if getattr(opt, 'neck_mod', 'none') == 'psa':
        active.append('psa')
    if getattr(opt, 'neck_mod', 'none') == 'gelan':
        active.append('gelan')
    return active


def _require_optional_decision(opt):
    report = _resolve_existing_path(getattr(opt, 'optional_decision', ''))
    if report is None:
        return '--optional-decision must point to a doc/REPORT/optional_decision_*.md file before optional experiments run'
    parts = [x.lower() for x in report.parts]
    if 'doc' not in parts or 'report' not in parts or not report.name.startswith('optional_decision_'):
        return '--optional-decision must point under doc/REPORT/ and start with optional_decision_'
    return None


def validate_structure_options(opt, parser=None):
    ensure_structure_option_defaults(opt)
    cfg = str(getattr(opt, 'cfg', '') or '').lower()
    cfg_name = Path(cfg).name
    message = None

    if 'fcos' in cfg_name and 'decode' in cfg_name:
        message = 'FCOS experiment cfg is decode-only in 1.3.6 and cannot be used by train.py/train_aux.py'

    if message is None and getattr(opt, 'p2_head', 'none') == 'anchor':
        if 'w6' not in cfg_name:
            message = '--p2-head anchor is supported only for W6 cfg files'
        elif 'p2' not in cfg_name:
            message = '--p2-head anchor requires a p2 cfg, e.g. cfg/training/yolov7-w6-p2.yaml'
    elif getattr(opt, 'p2_head', 'none') == 'fcos' and getattr(opt, 'det_head', 'anchor') == 'anchor':
        if 'w6' not in cfg_name:
            message = '--p2-head fcos is supported only for W6 cfg files'
        else:
            opt.det_head = 'hybrid'
            opt.anchor_free_levels = 'p2'

    if message is None and getattr(opt, 'neck_mod', 'none') == 'scdown':
        if 'w6' not in cfg_name:
            message = '--neck-mod scdown is supported only for W6 cfg files'
        elif 'scdown' not in cfg_name:
            message = '--neck-mod scdown requires a scdown cfg, e.g. cfg/training/yolov7-w6-scdown.yaml'
    elif message is None and getattr(opt, 'neck_mod', 'none') == 'psa':
        if 'w6' not in cfg_name:
            message = '--neck-mod psa is supported only for W6 cfg files'
        elif getattr(opt, 'psa_level', 'none') != 'p5':
            message = '--neck-mod psa only allows --psa-level p5 in 1.3.6'
        elif 'psa' not in cfg_name:
            message = '--neck-mod psa requires an experiment cfg, e.g. cfg/experiments/yolov7-w6-psa-p5.yaml'
    elif message is None and getattr(opt, 'neck_mod', 'none') == 'gelan':
        if 'w6' not in cfg_name:
            message = '--neck-mod gelan is supported only for W6 cfg files'
        elif 'gelan' not in cfg_name:
            message = '--neck-mod gelan requires an experiment cfg, e.g. cfg/experiments/yolov7-w6-gelan-neck.yaml'

    active = active_optional_experiments(opt)
    if message is None and len(active) > 1:
        message = f'optional experiments must run one at a time, got: {", ".join(active)}'
    decision_required = [x for x in active if x not in ('fcos', 'fcos_p2')]
    if message is None and decision_required:
        message = _require_optional_decision(opt)

    if message:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return opt


def validate_anchor_free_options(opt, parser=None):
    ensure_structure_option_defaults(opt)
    cfg_name = Path(str(getattr(opt, 'cfg', '') or '')).name.lower()
    message = None
    if getattr(opt, 'det_head', 'anchor') not in ('anchor', 'fcos', 'hybrid'):
        message = f'unsupported --det-head: {opt.det_head}'
    elif opt.det_head != 'anchor' and getattr(opt, 'head', 'coupled') == 'decoupled':
        message = '--det-head fcos|hybrid cannot be combined with --head decoupled in 1.4.1'
    elif opt.det_head == 'hybrid' and getattr(opt, 'anchor_free_levels', '') != 'p2':
        message = '--det-head hybrid is limited to --anchor-free-levels p2 in 1.4.1'
    elif opt.anchor_free_levels == 'p2' and 'w6' not in cfg_name:
        message = '--anchor-free-levels p2 is supported only for W6 cfg files'
    elif not (0.0 < float(opt.lambda_free) <= 2.0):
        message = '--lambda-free must be in (0, 2]'
    elif not (0.5 <= float(opt.fcos_center_radius) <= 3.0):
        message = '--fcos-center-radius must be between 0.5 and 3.0'
    elif getattr(opt, 'fcos_score_mode', 'sqrt_cls_centerness') not in ('sqrt_cls_centerness', 'mul_cls_centerness'):
        message = '--fcos-score-mode must be sqrt_cls_centerness or mul_cls_centerness'
    elif 'fcos-p2-decode' in cfg_name or 'decode' in cfg_name and 'fcos' in cfg_name:
        message = 'decode-only FCOS cfg cannot be used for train/test/export'
    if message:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return opt
