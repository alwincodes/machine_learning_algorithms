import matplotlib.pyplot as plt
def plot_graphs(X, y, prediction, cost_per_iter, xlabel, ylabel):
    _plot, axis = plt.subplots(1, 2, figsize= (10, 5))
    
    axis[0].set_title(f" {xlabel} / {ylabel}")
    axis[0].set_xlabel(xlabel)
    axis[0].set_ylabel(ylabel)
    axis[0].scatter(X, y, s=10, label="Training data")
    axis[0].plot(X, prediction, color="Orange", label="Fitted line (prediction)")
    axis[0].legend()

    axis[1].set_title("Gradient descent")
    axis[1].set_ylabel("Cost")
    axis[1].set_xlabel("Iterations")
    axis[1].plot(cost_per_iter, label="Cost")
    axis[1].legend()

    plt.show()