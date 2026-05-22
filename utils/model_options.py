from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def ensure_structure_option_defaults(opt):
    if not hasattr(opt, 'p2_head'):
        opt.p2_head = 'none'
    if not hasattr(opt, 'neck_mod'):
        opt.neck_mod = 'none'
    if not hasattr(opt, 'psa_level'):
        opt.psa_level = 'none'
    if not hasattr(opt, 'optional_decision'):
        opt.optional_decision = ''
    cfg_name = Path(str(getattr(opt, 'cfg', '') or '')).name.lower()
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
    if getattr(opt, 'p2_head', 'none') == 'fcos':
        active.append('fcos_p2')
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

    if 'fcos' in cfg_name:
        message = 'FCOS experiment cfg is decode-only in 1.3.6 and cannot be used by train.py/train_aux.py'

    if message is None and getattr(opt, 'p2_head', 'none') == 'anchor':
        if 'w6' not in cfg_name:
            message = '--p2-head anchor is supported only for W6 cfg files'
        elif 'p2' not in cfg_name:
            message = '--p2-head anchor requires a p2 cfg, e.g. cfg/training/yolov7-w6-p2.yaml'
    elif getattr(opt, 'p2_head', 'none') == 'fcos':
        if 'w6' not in cfg_name:
            message = '--p2-head fcos is supported only for W6 cfg files'
        else:
            message = '--p2-head fcos is decode-only in 1.3.6; use tools/decode_fcos_outputs.py and keep training on anchor P2'

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
    if message is None and active:
        message = _require_optional_decision(opt)

    if message:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return opt
