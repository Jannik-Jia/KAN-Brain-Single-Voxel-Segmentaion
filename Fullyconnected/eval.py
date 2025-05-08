#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估函数
"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模型评估函数
"""

import os
from utils.metrics import evaluate_model


# 为保持向后兼容性的包装函数
def evaluate_model_detailed(model, data_loader, device, result_path, dataset_name="test", class_names=None):
    """
    使用统一评估函数的包装，保持向后兼容性
    """
    return evaluate_model(
        model=model,
        data_loader=data_loader,
        device=device,
        result_path=result_path,
        dataset_name=dataset_name,
        class_names=class_names,
        detailed=True,
        plot=True
    )