import torch
import torch.nn as nn
import torch.nn.functional as F


def kd_loss(logits_student, logits_teacher, temperature):
    """
    实例级对齐损失 (Instance-level Alignment)
    标准KL散度损失，必须保持T²
    """
    loss_kd = F.kl_div(
        F.log_softmax(logits_student / temperature, dim=1),
        F.softmax(logits_teacher / temperature, dim=1),
        reduction='batchmean'
    )
    return loss_kd * (temperature ** 2)


def cc_loss(logits_student, logits_teacher, temperature):
    """
    类别级对齐损失 (Class-level Alignment)
    改进归一化：使用MSE而不是sum()÷C
    """
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    # 类别相关性矩阵：M = P^T @ P [C, C]
    student_matrix = torch.mm(pred_student.t(), pred_student)
    teacher_matrix = torch.mm(pred_teacher.t(), pred_teacher)

    # 改进：使用MSE归一化，数学上更合理
    return F.mse_loss(student_matrix, teacher_matrix, reduction='mean')


def bc_loss(logits_student, logits_teacher, temperature):
    """
    批次级对齐损失 (Batch-level Alignment)
    改进归一化：使用MSE而不是sum()÷B
    """
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)

    # Gram矩阵：G = P @ P^T [B, B]
    student_matrix = torch.mm(pred_student, pred_student.t())
    teacher_matrix = torch.mm(pred_teacher, pred_teacher.t())

    # 改进：使用MSE归一化，数学上更合理
    return F.mse_loss(student_matrix, teacher_matrix, reduction='mean')


class MultiLevelLogitDistillation(nn.Module):
    """
    改进的Multi-level Logit Distillation
    - 保持等权重（遵循原论文）
    - 改进CC/BC归一化（解决数值问题）
    - 对温度数量归一化（提高稳定性）
    """

    def __init__(self, temperatures=[2.0, 3.0, 4.0, 5.0, 6.0]):
        super().__init__()
        self.temperatures = temperatures

    def forward(self, logits_student, logits_teacher):
        """
        计算多层次logit蒸馏损失
        """
        total_kd_loss = 0.0
        total_cc_loss = 0.0
        total_bc_loss = 0.0

        logits_teacher = logits_teacher.detach()  # 确保教师logits不参与梯度计算

        # 对每个温度参数计算损失
        for temp in self.temperatures:
            total_kd_loss += kd_loss(logits_student, logits_teacher, temp)
            total_cc_loss += cc_loss(logits_student, logits_teacher, temp)
            total_bc_loss += bc_loss(logits_student, logits_teacher, temp)

        # 等权重求和，对温度数量归一化
        num_temps = len(self.temperatures)
        total_loss = (total_kd_loss + total_cc_loss + total_bc_loss) / num_temps

        return total_loss / 3