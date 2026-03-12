"""
==========================================================================
pcgrad.py — PCGrad（Project Conflicting Gradients）梯度冲突解决模块
==========================================================================
功能说明：
    实现论文 Section 3.4.1 "Gradient Match Strategy" 的 PCGrad 算法。
    
    核心问题：
        EditKG 有两个训练目标：
        1. 主任务损失 L_main = L_rec（推荐的 BPR 交叉熵损失）
        2. 辅助任务损失 L_aux = L_mmd（MMD 知识蒸馏损失）
        这两个损失的梯度可能冲突（方向相反），直接相加会互相抵消。
    
    解决方案（论文 Eq.25-28）：
        检测辅助任务梯度 g_aux 和主任务梯度 g_main 是否冲突：
        - 如果 g_aux · g_main < 0（冲突）：
          将 g_aux 投影到与 g_main 垂直的平面上，消除冲突分量
          g_aux = g_aux - (g_aux · g_main / ||g_main||²) * g_main
        - 如果 g_aux · g_main >= 0（不冲突）：
          保持原梯度

文件角色：
    被 main.py 创建 PCGrad 优化器包装类，在训练循环中调用 pc_backward()。
    
参考文献：
    Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020
==========================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class PCGrad():
    """
    PCGrad 优化器包装类
    
    将普通的 PyTorch 优化器包装后，提供 pc_backward() 方法来替代
    传统的 loss.backward() + optimizer.step()。
    
    使用方式（在 main.py 中）：
        optimizer = PCGrad(torch.optim.Adam(model.parameters(), lr=lr))
        optimizer.pc_backward([mmd_loss, batch_loss])  # [辅助损失, 主损失]
        optimizer.step()
    
    参数:
        optimizer: PyTorch 优化器实例（如 Adam）
        reduction: 梯度合并方式（'mean' 或 'sum'）
    """
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        """返回内部的 PyTorch 优化器"""
        return self._optim

    def zero_grad(self):
        """清零所有参数的梯度"""
        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        """使用修正后的梯度更新参数"""
        return self._optim.step()

    def pc_backward(self, objectives):
        """
        PCGrad 的核心接口：计算冲突解决后的梯度
        
        参数:
            objectives: 损失列表 [辅助任务损失, 主任务损失]
                        注意顺序：objectives[0] = 辅助任务（MMD损失）
                                   objectives[1] = 主任务（推荐损失）
        
        流程:
            1. _pack_grad(): 对每个损失分别反向传播，收集各自的梯度
            2. _project_conflicting(): 检测并解决梯度冲突（核心算法）
            3. _unflatten_grad(): 将一维梯度还原为参数形状
            4. _set_grad(): 将修正后的梯度设置到模型参数上
        """
        # 第一步：分别对每个目标反向传播，获取各自的梯度
        grads, shapes, has_grads = self._pack_grad(objectives)
        # 第二步：投影冲突梯度
        pc_grad = self._project_conflicting(grads, has_grads)
        # 第三步：将扁平化的梯度恢复为参数形状
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        # 第四步：设置修正后的梯度
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        """
        ★ 核心算法：投影冲突梯度，对应论文 Eq.25-28
        
        参数:
            grads: [辅助任务梯度(扁平化), 主任务梯度(扁平化)]
            has_grads: [辅助任务梯度掩码, 主任务梯度掩码]
        
        返回:
            merged_grad: 合并后的梯度（已解决冲突）
        
        算法步骤：
            1. shared = 两个任务都有梯度的参数位置（交集）
            2. 计算 g_aux · g_main（内积）
            3. 如果内积 < 0（冲突），投影辅助梯度：
               g_aux = g_aux - (g_aux · g_main / ||g_main||²) * g_main
               → 消除辅助梯度中与主梯度方向相反的分量
            4. 合并：共享参数位置取加权和，非共享位置直接求和
        """
        # 找出两个任务共享的参数（都有梯度的位置）
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        
        # grads[0] = 辅助任务梯度（MMD损失的梯度）
        # grads[1] = 主任务梯度（推荐损失的梯度）
        aux_task_grad = pc_grad[0]
        main_task_grad = pc_grad[1]
        
        # 计算辅助梯度和主梯度的内积（论文 Eq.25：检测冲突条件）
        am_grad_smi = torch.dot(aux_task_grad, main_task_grad)
        
        if am_grad_smi < 0:
            # ★ 梯度冲突！执行投影操作（论文 Eq.26-27）
            # g_aux = g_aux - (g_aux · g_main / ||g_main||²) * g_main
            # 含义：从辅助梯度中减去其在主梯度方向上的投影分量
            # 投影后的辅助梯度与主梯度正交，不再互相削弱
            aux_task_grad -= (am_grad_smi) * main_task_grad / (main_task_grad.norm()**2)
        
        new_pc_grad = [aux_task_grad, main_task_grad]

        # 合并梯度（论文 Eq.28）
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        # 共享参数：辅助梯度 + 主梯度（都乘以权重1.0）
        merged_grad[shared] = 1. * aux_task_grad[shared] + 1 * main_task_grad[shared]
        # 非共享参数：直接求和
        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in new_pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        """将修正后的梯度设置回模型的各个参数"""
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        """
        对每个目标函数分别反向传播，收集梯度
        
        参数:
            objectives: 损失列表
        
        返回:
            grads: 每个目标的梯度（扁平化为一维向量）
            shapes: 每个参数的形状（用于后续恢复）
            has_grads: 每个参数是否有梯度的掩码
        """
        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            # retain_graph=True：保留计算图，因为需要对多个目标分别反向传播
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        """将一维扁平化梯度恢复为各参数的原始形状"""
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = int(np.prod(shape))
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        """将所有参数的梯度拼接为一个一维向量"""
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        """
        从优化器管理的所有参数中收集当前梯度
        
        返回:
            grad: 各参数的梯度列表
            shape: 各参数的形状列表
            has_grad: 各参数是否有梯度的掩码列表
                      （有梯度的位置为1，无梯度的位置为0）
        """
        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                if p.grad is None:
                    # 该参数没有梯度（可能未参与当前目标的计算图）
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad
