import torch
import torch.nn.functional as F


def distillation_loss(args, student_outputs, teacher_outputs, T=4.0):
    if args.database == 'NYU-Depth-V2':
        pred = student_outputs
        tar = teacher_outputs.detach()
        student_probs_T = F.log_softmax(pred / T, dim=1)
        teacher_probs_T_0 = F.softmax(tar / T, dim=1)
        kl_loss = F.kl_div(student_probs_T, teacher_probs_T_0, reduction='batchmean') * (T * T) * 1/(pred.shape[2] * pred.shape[3])

    elif args.database == 'VGGSound-50k' and args.Tmodel == 'CPSP':
        teacher_outputs = teacher_outputs[1].detach()
        student_probs_T = F.log_softmax(student_outputs / T, dim=1)
        teacher_probs_T = F.softmax(teacher_outputs / T, dim=1)
        kl_loss = F.kl_div(student_probs_T, teacher_probs_T, reduction='batchmean') * (T * T)

    else:
        teacher_outputs = teacher_outputs.detach()
        student_probs_T = F.log_softmax(student_outputs / T, dim=1)
        teacher_probs_T = F.softmax(teacher_outputs / T, dim=1)
        kl_loss = F.kl_div(student_probs_T, teacher_probs_T, reduction='batchmean') * (T * T)

    return kl_loss


if __name__ == '__main__':
    pred = torch.randn(32, 10)
    target = torch.randn(32, 10)

    kl_loss = distillation_loss(pred, target)
    print(kl_loss)