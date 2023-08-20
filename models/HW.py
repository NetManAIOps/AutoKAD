from statsmodels.tsa.holtwinters import ExponentialSmoothing as HW
import pandas as pd


def predict(test_seq: pd.Series, params):
    # print(params)
    model = HW(test_seq, **params).fit()

    predicted = model.predict(0, len(test_seq)-1).to_numpy()

    return predicted


def train(train_seq, params):
    model = HW(train, **params).fit()

    return model


def test(model, test_seq, params):
    return model.predict(0, len(test_seq)-1).to_numpy()



def train_and_predict(train_seq, test_seq, params):
    predicted = predict(test_seq, params)

    return predicted


if __name__ == '__main__':
    df = pd.read_csv("../data/tzs/test/e0747cad-8dc8-38a9-a9ab-855b61f5551d.csv")
    print(df)

    predicted = predict(df['value'], None)

    print(predicted)
    print(predicted.shape)
