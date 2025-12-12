class SchemaValidator:
    """根据 Family 检查 Experiment JSON 的完整性"""

    # 定义每个 Family 必须包含的 details 字段
    FAMILY_REQUIREMENTS = {
        "mlp": [
            "input_dim", "output_dim", "hidden_layers", "activation", 
            "dropout", "residual", "attention", "feature_interaction"
        ],
        "kan": [
            "kan_config", "input_dim", "output_dim"
        ],
        "tabnet": [
            "tabnet_config", "input_dim", "output_dim", "dropout"
        ],
        "pseudo_inverse": [
            "pseudo_inverse_config", "input_dim", "output_dim"
        ],
        "classical_ml": [
            "classical_ml_config", "input_dim", "output_dim"
        ]
    }

    @staticmethod
    def validate(experiment_dict):
        """
        验证实验字典
        Returns: (bool, list_of_errors)
        """
        errors = []
        
        # 1. 检查顶层必须字段
        required_top = ["experiment_id", "method_name", "method_key", "family", "subfamily", "results"]
        for field in required_top:
            if field not in experiment_dict:
                errors.append(f"Missing top-level field: {field}")

        # 2. 检查 Family 特异性字段
        family = experiment_dict.get("family")
        details = experiment_dict.get("model", {}).get("details", {})
        
        if family in SchemaValidator.FAMILY_REQUIREMENTS:
            reqs = SchemaValidator.FAMILY_REQUIREMENTS[family]
            for field in reqs:
                if field not in details or details[field] is None:
                    # 允许 false/0，但不允许 None 或缺失
                    errors.append(f"Family '{family}' requires 'model.details.{field}'")
        else:
            errors.append(f"Unknown family: {family}. Please update Validator.")

        return len(errors) == 0, errors