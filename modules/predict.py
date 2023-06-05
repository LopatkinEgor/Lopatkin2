import json
import os
import dill
import pandas as pd

path = os.environ.get('PROJECT_PATH', '.')


def predict():
    data1 = []
    fds1 = sorted(os.listdir(f'{path}/data/models/'))

    with open(f'data/models/{fds1[-1]}', 'rb') as file:
        model = dill.load(file)

    fds2 = sorted(os.listdir(f'{path}/data/test/'))
    for i in fds2:
        data = []
        with open(f'{path}/data/test/{i}') as fin:
            form = json.load(fin)

        df = pd.DataFrame.from_dict([form])
        y = model.predict(df)

        data.append(i[:10])
        data.append(y)
        data1.append(data)
    df1 = pd.DataFrame(data1, columns=['файл','предикт'])
    df1.to_csv(f'{path}/data/predictions/preds_{fds1[-1][10:22]}.csv', index=False)


if __name__ == '__main__':
    predict()
