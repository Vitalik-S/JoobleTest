from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class Scaler(ABC):

    @abstractmethod
    def transform(self, test_data: np.ndarray, train_data: np.ndarray) -> np.ndarray:
        pass


class ZScoreScaler(Scaler):

    def transform(self, test_data: np.ndarray, train_data: np.ndarray) -> np.ndarray:
        mean_features = train_data.mean(axis=0)
        std_features = train_data.std(axis=0)
        z_data_features = (test_data - mean_features) / std_features
        return z_data_features


class JobFeaturePreprocess:

    def __init__(self, train_data_path: str, test_data_path: str, scaler: Scaler):
        self._scaler = scaler
        self._train_data_path = train_data_path
        self._test_data_path = test_data_path

    @property
    def scaler(self) -> Scaler:
        return self._scaler

    @scaler.setter
    def scaler(self, scaler):
        self._scaler = scaler

    def preprocess_data(self) -> pd.DataFrame:
        #При использовании модуля в будущем планируется добавление факторов новых типов (в нашем тесте только признаки типа “2”)
        #Если я правильно понял этот пункт, то добавление нового типа означает, что может быть другой файл где будет только тип "3" - 512 float чисел к примеру
        test_data = pd.read_csv(self._test_data_path, sep='\t', chunksize=100000)
        test_data = pd.concat(test_data)
        test_data_np = np.delete(np.array(test_data['features'].str.split(',', expand=True).astype(float)), 0, 1)

        test_data = test_data.drop(['features'], axis=1)

        train_data = pd.read_csv(self._train_data_path, sep='\t', chunksize=100000)
        train_data = pd.concat(train_data)
        train_data_np = np.delete(np.array(train_data['features'].str.split(',', expand=True).astype(float)), 0, 1)
        # 2
        z_data_features = self._scaler.transform(test_data_np, train_data_np)
        test_data[['feature_2_stand_' + str(i) for i in range(len(z_data_features[0]))]] = z_data_features
        # 3
        max_i_features = test_data_np.argmax(axis=1)
        test_data['max_feature_2_index'] = max_i_features
        # 4
        mean_features = test_data_np.mean(axis=0)
        max_feature_2_abs_mean_diff = [np.abs(test_data_np[i][max_i_features[i]] - mean_features[max_i_features[i]]) for i in range(len(max_i_features))]
        test_data['max_feature_2_abs_mean_diff'] = max_feature_2_abs_mean_diff

        return test_data
