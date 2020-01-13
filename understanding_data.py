import numpy as np  # 1.16.4
import pandas as pd  # 0.24.2
import matplotlib # 2.2.4
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-0.005 * x))


def sigmoid_derivative(x):
    return 0.005 * x * (1 - x)

def read_and_divide_into_train_and_test(csv_file):

    df=pd.read_csv(csv_file)
    df=df.replace("?",np.nan)
    df=df.dropna(axis=0,how="any")

    cols = df.columns
    for col in cols:
        df[col] = df[col].astype(int)

    clas=np.array(df["Class"])
    total_row=df.shape[0]
    test_inputs = df.tail(round(total_row * 1 / 5))
    training_inputs=df.head(round(total_row*8/10))

    training_labels = []
    for i in training_inputs["Class"]:
        training_labels.append([i])

    test_labels = np.array(pd.DataFrame(test_inputs["Class"]))

    training_inputs=training_inputs.drop(['Code_number','Class'],axis=1)

    test_inputs = test_inputs.drop(['Code_number', 'Class'], axis=1)
    df=df.drop(['Code_number','Class'],axis=1)
    corraption=df.corr()


    figure, cax = plt.subplots()
    im = cax.imshow(corraption.values)

    plt.xticks(range(len(df.columns)), df.columns, rotation=90)
    plt.yticks(range(len(df.columns)), df.columns)
    cax.set_facecolor((0, 0, 0))

    plt.colorbar(im, ax=cax)
    for i in range(len(corraption.columns)):
        for j in range(len(corraption.columns)):
            text = cax.text(j, i, np.around(corraption.iloc[i, j], decimals=3),
                           ha="center", va="center", color="white")
    plt.show()
    return   training_inputs, training_labels, test_inputs, test_labels


def run_on_test_set(test_inputs, test_labels, weights):
    tp = 0

    test_outputs=sigmoid(np.dot(test_inputs,weights))
    test_predictions=[]
    for i in test_outputs:
        if(i>0.5):
            test_predictions.append(1)
        else:
            test_predictions.append(0)
    cnt = 0
    for predicted_val, label in zip(test_predictions, test_labels):
        cnt += 1
        if predicted_val == label:
           tp += 1
    accuracy = tp / cnt
    return accuracy


def plot_loss_accuracy(accuracy_array, loss_array):
    x = plt
    x.subplot(2, 1, 1)
    x.ylabel("Accuracy")
    x.plot(accuracy_array, "b-")

    x.subplot(2, 1, 2)
    x.ylabel("Loss")
    x.plot(loss_array, "r-")
    x.show()


def main():
    csv_file = './breast-cancer-wisconsin.csv'
    iteration_count = 2500
    np.random.seed(1)
    weights = 2 * np.random.random((9, 1)) - 1
    accuracy_array = []
    loss_array = []
    training_inputs, training_labels, test_inputs, test_labels = read_and_divide_into_train_and_test(csv_file)

    for iteration in range(iteration_count):

        outputs = np.dot(training_inputs,weights)

        outputs = sigmoid(outputs)

        loss = training_labels - outputs

        tuning = loss*sigmoid_derivative(outputs)

        weights += np.dot(np.transpose(training_inputs),tuning)

        loss = loss.mean(axis=0)
        loss_array.append(loss[0])
        accuracy_array.append(run_on_test_set(test_inputs,test_labels,weights))

    plot_loss_accuracy(accuracy_array, loss_array)


if __name__ == '__main__':
    main()
