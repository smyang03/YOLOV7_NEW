import torch


class WIoUState:
    def __init__(self, momentum=0.01, initial_mean=1.0, device=None):
        self.momentum = float(momentum)
        self.running_mean = torch.tensor(float(initial_mean), device=device)

    def to(self, device):
        self.running_mean = self.running_mean.to(device)
        return self

    def update(self, iou_loss):
        with torch.no_grad():
            value = iou_loss.detach().mean().clamp(min=1e-6)
            self.running_mean.mul_(1.0 - self.momentum).add_(value * self.momentum)

    def state_dict(self):
        return {
            'momentum': self.momentum,
            'running_mean': float(self.running_mean.detach().cpu()),
        }

    def load_state_dict(self, state):
        if not state:
            return
        self.momentum = float(state.get('momentum', self.momentum))
        value = float(state.get('running_mean', self.running_mean.detach().cpu()))
        self.running_mean = self.running_mean.new_tensor(value)


def wiou_v3_loss(iou, state=None, eps=1e-7):
    iou_loss = (1.0 - iou.clamp(0.0, 1.0)).clamp(min=0.0)
    if state is None:
        return iou_loss

    state.to(iou.device)
    if torch.is_grad_enabled():
        state.update(iou_loss)
    beta = (iou_loss.detach() / (state.running_mean + eps)).clamp(min=eps, max=6.0)
    focus = beta / (2.0 * torch.pow(torch.tensor(1.9, device=iou.device), beta - 2.0) + eps)
    return iou_loss * focus
