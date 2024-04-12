# encoding: utf-8
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class VehicleDynamicsModel(nn.Module):
    def __init__(self, input_size=10, output_size=5):
        super(VehicleDynamicsModel, self).__init__()
        # 定义第一个隐藏层
        self.hidden1 = nn.Linear(input_size, 50, bias=True)
        self.active1 = nn.ReLU()
        # 定义第一个隐藏层
        self.hidden2 = nn.Linear(50, 50)
        self.active2 = nn.ReLU()
        # 定义预测回归层
        self.hidden3 = nn.Linear(50, 10)
        self.active3 = nn.ReLU()
        self.regression = nn.Linear(10, output_size)

    # 定义网络的向前传播路径
    def forward(self, x):
        x = self.hidden1(x)
        x = self.active1(x)
        x = self.hidden2(x)
        x = self.active2(x)
        x = self.hidden3(x)
        x = self.active3(x)
        output = self.regression(x)
        # 输出为output
        return output


class VehicleDynamicsLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=100, output_size=5, num_layers=2):
        super(VehicleDynamicsLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # 前向传播 LSTM
        out, _ = self.lstm(x, (h0, c0))  # out的形状：(batch_size, seq_length, hidden_size)

        # 仅使用最后一个时间步的输出
        out = self.fc(out[:, -1, :])

        return out


def data_process():
    # 1. 生成训练数据 tstep = 8 ms
    path = "/home/plusai/Documents/external_projects/NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/data_to_train_0.csv"
    df = pd.read_csv(path)
    length = len(df)
    input_column = [
        "vx_mps",
        "vy_mps",
        "dpsi_radps",
        "ax_mps2",
        "ay_mps2",
        "deltawheel_rad",
        "TwheelRL_Nm",
        "TwheelRR_Nm",
        "pBrakeF_bar",
        "pBrakeR_bar",
    ]
    output_column = [
        "vx_mps",
        "vy_mps",
        "dpsi_radps",
        "ax_mps2",
        "ay_mps2",
    ]
    input_data = df.loc[:length - 2, input_column]  # 输入特征
    output_data = df.loc[1:, output_column]  # 输出特征
    input_data = input_data.to_numpy()
    print("input_data shape: {}".format(input_data.shape))
    output_data = output_data.to_numpy()
    print("output_data shape: {}".format(output_data.shape))

    scaler = MinMaxScaler()
    input_data = scaler.fit_transform(input_data)
    output_data = scaler.fit_transform(output_data)

    return input_data, output_data


def plot(args, predicted_output, test_output_data):
    test_length = predicted_output.shape[0]
    x_list = np.arange(test_length)
    fig = plt.figure(figsize=(20, 20))
    fig.subplots_adjust(hspace=0.186, wspace=0.1, left=0.055, bottom=0.055, right=0.96, top=0.96)

    ax = fig.add_subplot(5, 1, 1)
    ax.set_xlabel("index")
    ax.set_ylabel("vx_mps")
    ax.plot(x_list, predicted_output[:, 0].T, label="vx_mps_predict")
    ax.plot(x_list, test_output_data[:, 0].T, label="vx_mps_gt")
    ax.legend()

    ax = fig.add_subplot(5, 1, 2)
    ax.set_xlabel("index")
    ax.set_ylabel("vy_mps")
    ax.plot(x_list, predicted_output[:, 1].T, label="vy_mps_predict")
    ax.plot(x_list, test_output_data[:, 1].T, label="vy_mps_gt")
    ax.legend()

    ax = fig.add_subplot(5, 1, 3)
    ax.set_xlabel("index")
    ax.set_ylabel("dpsi_radps")
    ax.plot(x_list, predicted_output[:, 4].T, label="dpsi_radps_predict")
    ax.plot(x_list, test_output_data[:, 4].T, label="dpsi_radps_gt")
    ax.legend()

    ax = fig.add_subplot(5, 1, 4)
    ax.set_xlabel("index")
    ax.set_ylabel("ax_mps2")
    ax.plot(x_list, predicted_output[:, 4].T, label="ax_mps2_predict")
    ax.plot(x_list, test_output_data[:, 4].T, label="ax_mps2_gt")
    ax.legend()

    ax = fig.add_subplot(5, 1, 5)
    ax.set_xlabel("index")
    ax.set_ylabel("ay_mps2")
    ax.plot(x_list, predicted_output[:, 4].T, label="ay_mps2_predict")
    ax.plot(x_list, test_output_data[:, 4].T, label="ay_mps2_gt")
    ax.legend()

    if args.savepic:
        pic_path = os.path.join(args.output, "model/test_model.png")
        plt.savefig(pic_path)
    plt.show()
    plt.close()


def model_evaluation(args):
    input_data, output_data = data_process()
    test_input_tensor = torch.from_numpy(input_data.astype(np.float32))
    test_output_tensor = torch.from_numpy(output_data.astype(np.float32))

    # 2. deep learning model
    # 加载路径
    load_path = os.path.join(args.output, "model/model_checkpoint.pth")
    # 加载模型
    checkpoint = torch.load(load_path)
    # 重新构建模型
    model = VehicleDynamicsModel()
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    with torch.no_grad():
        model.eval()
        predicted_output_tensor = model(test_input_tensor)

    criterion = nn.MSELoss()
    loss = criterion(predicted_output_tensor, test_output_tensor)
    print("Mean Squared Error:", loss.item())

    test_length = 3000
    plot_predicted_output = predicted_output_tensor[:test_length]
    plot_predicted_output = plot_predicted_output.numpy()
    plot_output_data = test_output_tensor[:test_length]
    plot_output_data = plot_output_data.numpy()
    plot(args, plot_predicted_output, plot_output_data)


def model_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设备
    print("device: {}".format(device))

    input_data, output_data = data_process()
    x_train, x_test, y_train, y_test = train_test_split(
        input_data, output_data, test_size=0.1, shuffle=False
    )

    x_train = torch.from_numpy(x_train.astype(np.float32)).to(device)
    y_train = torch.from_numpy(y_train.astype(np.float32)).to(device)
    x_test = torch.from_numpy(x_test.astype(np.float32)).to(device)
    y_test = torch.from_numpy(y_test.astype(np.float32)).to(device)

    train_data = Data.TensorDataset(x_train, y_train)
    test_data = Data.TensorDataset(x_test, y_test)

    batch_size = 20000
    train_loader = Data.DataLoader(
        dataset=train_data,  # 使用的数据集
        batch_size=batch_size,
        shuffle=False,  # 每次迭代前打乱数据
    )

    test_loader = Data.DataLoader(
        dataset=test_data,  # 使用的数据集
        batch_size=1000,
        shuffle=False,  # 每次迭代前打乱数据
    )

    # 2. deep learning model
    model = VehicleDynamicsModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # 3. training process
    num_epochs = 150

    print("start to taining model")
    for epoch in tqdm.trange(num_epochs):
        loss_list = []
        for inputs, outputs_gt in train_loader:
            inputs = inputs.to(device)
            outputs_gt = outputs_gt.to(device)
            # 前向传播
            predicted_output = model(inputs)

            # 计算损失
            loss = criterion(predicted_output, outputs_gt)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        if (epoch + 1) % 5 == 0:
            print(
                "Epoch [{}/{}], Loss: {:.4f}".format(
                    epoch + 1, num_epochs, np.mean(loss_list)
                )
            )

    # 保存模型参数和其他信息
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        # 如果有需要的话，还可以保存其他信息，比如模型的超参数等
    }
    # 选择保存路径
    save_path = os.path.join(args.output, "model/model_checkpoint.pth")
    # 保存模型
    torch.save(checkpoint, save_path)

    # 4. test process
    test_length = 3000
    test_input_data = x_test[:test_length]
    test_output_data = y_test[:test_length]
    with torch.no_grad():
        predicted_output = model(test_input_data)
        predicted_output = predicted_output.to("cpu").numpy()
    test_output_data = test_output_data.to("cpu").numpy()
    plot(args, predicted_output, test_output_data)


def main(args):
    if args.train:
        model_training(args)
    else:
        model_evaluation(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("run_vehicle_model")
    parser.add_argument(
        "--output",
        "-o",
        default=os.path.dirname(os.path.abspath(__file__)),
        type=str,
        help="Abspath of folder to put all output files in. {} as default.".format(
            os.path.dirname(os.path.abspath(__file__))
        ),
    )
    parser.add_argument(
        "--train", default=False, action="store_true", help="Whether to train"
    )
    parser.add_argument(
        "--savepic",
        default=False,
        action="store_true",
        help="Whether to generate data plot",
    )
    args = parser.parse_args()
    main(args)
