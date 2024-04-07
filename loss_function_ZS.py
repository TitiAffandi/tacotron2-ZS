# adapted from https://github.com/NVIDIA/tacotron2/blob/master/loss_function.py
# add some modifications

from torch import nn

class Tacotron2Loss(nn.Module):
    def __init__(self):
        super(Tacotron2Loss, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target, speaker_embed_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        speaker_embed_target.requires_grad = False

        mel_out, mel_out_postnet, gate_out, _, speaker_embed_out = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        return mel_loss + gate_loss

class Tacotron2Loss1(nn.Module):
    def __init__(self):
        super(Tacotron2Loss1, self).__init__()

    def forward(self, model_output, targets):
        mel_target, gate_target, speaker_embed_target = targets[0], targets[1], targets[2]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        speaker_embed_target.requires_grad = False

        mel_out, mel_out_postnet, gate_out, _, speaker_embed_out = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + \
            nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)
        speaker_embed_loss = nn.MSELoss()(speaker_embed_out, speaker_embed_target)
        return mel_loss + gate_loss + speaker_embed_loss
