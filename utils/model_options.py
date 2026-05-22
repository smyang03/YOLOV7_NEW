from pathlib import Path


def ensure_structure_option_defaults(opt):
    if not hasattr(opt, 'p2_head'):
        opt.p2_head = 'none'
    if not hasattr(opt, 'neck_mod'):
        opt.neck_mod = 'none'
    cfg_name = Path(str(getattr(opt, 'cfg', '') or '')).name.lower()
    if opt.p2_head == 'none' and 'w6' in cfg_name and 'p2' in cfg_name:
        opt.p2_head = 'anchor'
    if opt.neck_mod == 'none' and 'w6' in cfg_name and 'scdown' in cfg_name:
        opt.neck_mod = 'scdown'
    return opt


def validate_structure_options(opt, parser=None):
    ensure_structure_option_defaults(opt)
    cfg = str(getattr(opt, 'cfg', '') or '').lower()
    cfg_name = Path(cfg).name
    message = None

    if getattr(opt, 'p2_head', 'none') == 'anchor':
        if 'w6' not in cfg_name:
            message = '--p2-head anchor is supported only for W6 cfg files'
        elif 'p2' not in cfg_name:
            message = '--p2-head anchor requires a p2 cfg, e.g. cfg/training/yolov7-w6-p2.yaml'

    if message is None and getattr(opt, 'neck_mod', 'none') == 'scdown':
        if 'w6' not in cfg_name:
            message = '--neck-mod scdown is supported only for W6 cfg files'
        elif 'scdown' not in cfg_name:
            message = '--neck-mod scdown requires a scdown cfg, e.g. cfg/training/yolov7-w6-scdown.yaml'

    if message:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)
    return opt
