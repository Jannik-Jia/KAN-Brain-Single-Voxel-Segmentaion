# Alex完全一致的模型定义
class AlexIdenticalModel(nn.Module):
    def __init__(self, input_dim=42, num_classes=52):
        super(AlexIdenticalModel, self).__init__()
        # 完全对应Alex的结构：无BatchNorm
        self.fc1 = nn.Linear(input_dim, 4096)
        self.fc2 = nn.Linear(4096, 4096) 
        self.fc3 = nn.Linear(4096, 4096)
        self.fc4 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 严格按照Alex的结构
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.dropout(F.relu(self.fc4(x)))
        x = self.fc5(x)  # 无激活函数
        return x

# 训练配置调整
config = {
    'batch_size': 128,        # 改为Alex的批大小
    'learning_rate': 0.00001, # 一致
    'weight_decay': 0.00001,  # 一致
    'num_epochs': 25,         # 一致
    'dropout_rate': 0.5,      # 一致
    'hidden_dim': 4096,       # 一致
    'num_hidden_layers': 4,   # 一致
}