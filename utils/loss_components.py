import torch
import torch.nn.functional as F

from utils.wiou import WIoUState, wiou_v3_loss


DEFAULT_LOSS_OPTIONS = {
    'head': 'coupled',
    'loss_box': 'ciou',
    'assign': 'simota',
    'loss_cls': 'bce',
}


def ensure_loss_option_defaults(opt):
    for key, value in DEFAULT_LOSS_OPTIONS.items():
        if not hasattr(opt, key):
            setattr(opt, key, value)
    return opt


def validate_loss_options(opt, parser=None):
    ensure_loss_option_defaults(opt)
    message = None
    if opt.loss_cls == 'vfl' and opt.assign != 'tal':
        message = '--loss-cls vfl requires --assign tal'
    if message:
        if parser is not None:
            parser.error(message)
        raise ValueError(message)


def apply_loss_options(hyp, opt):
    ensure_loss_option_defaults(opt)
    hyp['head'] = opt.head
    hyp['loss_box'] = opt.loss_box
    hyp['assign'] = opt.assign
    hyp['loss_cls'] = opt.loss_cls
    hyp.setdefault('tal_topk', 10)
    hyp.setdefault('tal_alpha', 1.0)
    hyp.setdefault('tal_beta', 6.0)
    hyp.setdefault('wiou_momentum', 0.01)
    hyp.setdefault('small_iou_weight', 0.0)
    return hyp


def init_loss_options(loss_obj, hyp, device):
    loss_obj.head = hyp.get('head', 'coupled')
    loss_obj.loss_box = hyp.get('loss_box', 'ciou')
    loss_obj.assign = hyp.get('assign', 'simota')
    loss_obj.loss_cls = hyp.get('loss_cls', 'bce')
    loss_obj.last_positive_count = 0
    loss_obj.wiou_state = WIoUState(hyp.get('wiou_momentum', 0.01), device=device) \
        if loss_obj.loss_box == 'wiou_v3' else None


def box_loss_from_iou(iou, loss_obj):
    if getattr(loss_obj, 'loss_box', 'ciou') == 'wiou_v3':
        return wiou_v3_loss(iou, getattr(loss_obj, 'wiou_state', None))
    return 1.0 - iou


def varifocal_loss(pred_score, target_score, alpha=0.75, gamma=2.0):
    pred_sigmoid = pred_score.sigmoid()
    weight = alpha * pred_sigmoid.pow(gamma) * (target_score <= 0.0).float()
    weight += target_score * (target_score > 0.0).float()
    loss = F.binary_cross_entropy_with_logits(pred_score, target_score, reduction='none') * weight
    return loss.mean()


def classification_loss(loss_obj, pred_score, class_index, iou):
    if getattr(loss_obj, 'loss_cls', 'bce') != 'vfl':
        target = torch.full_like(pred_score, loss_obj.cn, device=pred_score.device)
        target[range(pred_score.shape[0]), class_index] = loss_obj.cp
        return loss_obj.BCEcls(pred_score, target)

    target_score = torch.zeros_like(pred_score, device=pred_score.device)
    target_score[range(pred_score.shape[0]), class_index] = iou.detach().clamp(0.0, 1.0).type(target_score.dtype)
    return varifocal_loss(pred_score, target_score)


def get_loss_positive_count(loss_obj):
    return int(getattr(loss_obj, 'last_positive_count', 0))


def build_loss_state(opt, loss_obj):
    ensure_loss_option_defaults(opt)
    wiou_state = getattr(loss_obj, 'wiou_state', None)
    return {
        'wiou': wiou_state.state_dict() if wiou_state else None,
        'assign': opt.assign,
        'loss_box': opt.loss_box,
        'loss_cls': opt.loss_cls,
        'head': opt.head,
    }


def load_loss_state(loss_obj, state, logger=None):
    if not state:
        if getattr(loss_obj, 'wiou_state', None) is not None and logger is not None:
            logger.warning('WIoU state not found in checkpoint; initialized a new WIoU running state.')
        return
    wiou_state = getattr(loss_obj, 'wiou_state', None)
    if wiou_state is not None:
        if state.get('wiou'):
            wiou_state.load_state_dict(state['wiou'])
        elif logger is not None:
            logger.warning('WIoU state missing in checkpoint loss_state; initialized a new WIoU running state.')
