import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model1.pth'):
        model_folder_path = './model1'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
    def load(self, file_name='model1.pth'):
        model_folder_path = './model1'
        file_name = os.path.join(model_folder_path, file_name)
        self.load_state_dict(torch.load(file_name))
        self.eval()  # Set the model to evaluation mode


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.01)
        self.criterion = nn.MSELoss()
    '''
    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred = self.model(state)



        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            # print(f'reward= {Q_new}')
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    '''
    '''

        for idx in range(len(done)):
            Q_new = reward[idx]
            # print('QB', Q_new)
            if not done[idx]:

                # Compute the maximum Q value for the next state
                max_future_q = torch.max(self.model(next_state[idx])).item()  # Extract scalar
                Q_new = reward[idx] + (self.gamma) * max_future_q  # Calculate new Q value as scalar
                # print('QA', Q_new)
            
            # Ensure action[idx] is a scalar
            action_idx = action[idx].item() if action[idx].dim() == 0 else torch.argmax(action[idx]).item()
            # Update the target value for the specific action
            # print('targetBE', target[idx] )
            target[idx][action_idx] = Q_new
            # print('targetAF', target[idx] )

    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        # print('target',target )

        loss.backward()

        self.optimizer.step()
            '''

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # score = torch.tensor(score, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )
            # score = torch.unsqueeze(score, 0)

        # 1: predicted Q values with current state
        pred = self.model(state)
        target = pred.clone()
        Q_new = 0
        for idx in range(len(done)):
            Q_new = reward[idx] 
            # print(f'reward= {Q_new}')
            if not done[idx]:
                Q_new = Q_new + self.gamma * torch.max(self.model(next_state[idx]))  # TODO

            target[idx][torch.argmax(action[idx]).item()] = Q_new
            

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        # print('target',target )

        loss.backward()

        self.optimizer.step()

