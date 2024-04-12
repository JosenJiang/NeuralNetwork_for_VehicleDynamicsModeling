import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    path = "/home/plusai/Documents/external_projects/NeuralNetwork_for_VehicleDynamicsModeling/inputs/trainingdata/data_to_train_0.csv"
    df = pd.read_csv(path)
    column_names = df.columns.tolist()
    print("column names: {}".format(column_names))
    length = len(df["vx_mps"])
    print("data length: {}".format(length))

    end = 1000
    df = df.iloc[:end]
    x_list = np.arange(end)

    df["vx_mps_dot"] = df["vx_mps"].diff() / 0.008
    df["vy_mps_dot"] = df["vy_mps"].diff() / 0.008

    fig = plt.figure(figsize=(20, 40))
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    ax = fig.add_subplot(2, 2, 1)
    ax.set_xlabel("index")
    ax.set_ylabel("vx_mps")
    ax.plot(x_list, df["vx_mps"], label="vx_mps")
    ax.legend()

    ax = fig.add_subplot(2, 2, 2)
    ax.set_xlabel("index")
    ax.set_ylabel("vy_mps")
    ax.plot(x_list, df["vy_mps"], label="vy_mps")
    ax.legend()

    ax = fig.add_subplot(2, 2, 3)
    ax.set_xlabel("index")
    ax.set_ylabel("ax_mps2")
    ax.plot(x_list, df["ax_mps2"], label="ax_mps2")
    ax.plot(x_list, df["vx_mps_dot"], label="vx_mps_dot")
    ax.legend()

    ax = fig.add_subplot(2, 2, 4)
    ax.set_xlabel("index")
    ax.set_ylabel("ay_mps2")
    ax.plot(x_list, df["ay_mps2"], label="ay_mps2")
    ax.plot(x_list, df["vy_mps_dot"], label="vy_mps_dot")
    ax.legend()

    plt.show()


if __name__ == "__main__":
    main()
