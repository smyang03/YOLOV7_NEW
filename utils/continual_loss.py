import torch
import torch.nn as nn
import torch.nn.functional as F


def parse_float_or_schedule(value):
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ':' in text:
        start, end = text.split(':', 1)
        return float(start), float(end)
    return float(text)


def scheduled_value(value, epoch=0, epochs=1):
    if isinstance(value, tuple):
        start, end = value
        if epochs <= 1:
            return float(end)
        ratio = min(max(float(epoch) / float(max(epochs - 1, 1)), 0.0), 1.0)
        return float(start + (end - start) * ratio)
    return float(value)


def max_schedule_value(value):
    return max(value) if isinstance(value, tuple) else float(value)


def normalize_outputs(outputs):
    if isinstance(outputs, tuple) and len(outputs) >= 2 and isinstance(outputs[1], (list, tuple)):
        outputs = outputs[1]
    if isinstance(outputs, torch.Tensor):
        return [outputs]
    if isinstance(outputs, (list, tuple)):
        tensors = []
        for item in outputs:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
        return tensors
    raise TypeError(f'unsupported distillation output type: {type(outputs)}')


def align_output_lists(student, teacher):
    if len(student) == len(teacher) * 2:
        student = student[:len(teacher)]
    if len(student) != len(teacher):
        raise ValueError(f'distill output count mismatch: student={len(student)} teacher={len(teacher)}')
    return student, teacher


class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.0, beta=0.0, conf_thres=0.5):
        super().__init__()
        self.alpha = parse_float_or_schedule(alpha)
        self.beta = parse_float_or_schedule(beta)
        self.conf_thres = float(conf_thres)

    def active(self):
        return max_schedule_value(self.alpha) > 0 or max_schedule_value(self.beta) > 0

    def forward(self, student_outputs, teacher_outputs, epoch=0, epochs=1):
        alpha = scheduled_value(self.alpha, epoch, epochs)
        beta = scheduled_value(self.beta, epoch, epochs)
        student = normalize_outputs(student_outputs)
        teacher = normalize_outputs(teacher_outputs)
        student, teacher = align_output_lists(student, teacher)

        device = student[0].device if student else torch.device('cpu')
        cls_loss = torch.zeros((), device=device)
        reg_loss = torch.zeros((), device=device)
        checked = 0
        for s, t in zip(student, teacher):
            if tuple(s.shape[:-1]) != tuple(t.shape[:-1]):
                raise ValueError(f'distill output shape mismatch: student={tuple(s.shape)} teacher={tuple(t.shape)}')
            if s.shape[-1] < t.shape[-1]:
                raise ValueError(
                    f'distill class channel mismatch: student last dim {s.shape[-1]} < teacher last dim {t.shape[-1]}')
            checked += 1
            t = t.detach().to(device=device, dtype=s.dtype)
            if s.shape[-1] >= 5:
                if alpha > 0:
                    cls_loss = cls_loss + F.mse_loss(s[..., 4:t.shape[-1]], t[..., 4:])
                if beta > 0:
                    mask = t[..., 4] >= self.conf_thres
                    if mask.any():
                        reg_loss = reg_loss + F.mse_loss(s[..., :4][mask], t[..., :4][mask])
            else:
                if alpha > 0:
                    cls_loss = cls_loss + F.mse_loss(s, t)

        if checked:
            cls_loss = cls_loss / checked
            reg_loss = reg_loss / checked
        total = cls_loss * alpha + reg_loss * beta
        items = {
            'distill_total': float(total.detach().cpu()) if total.numel() else 0.0,
            'distill_cls': float(cls_loss.detach().cpu()) if cls_loss.numel() else 0.0,
            'distill_reg': float(reg_loss.detach().cpu()) if reg_loss.numel() else 0.0,
            'alpha': alpha,
            'beta': beta,
        }
        return total, items
