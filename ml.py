import logging
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

DATA_FOLDER = "data"
FILE_NAME = "ObesityDataSet.csv"


def stratified_split_train_test(df: pd.DataFrame, target_column: str, test_size: float = 0.2, random_state: int = 42):
    """
    Split the DataFrame into training and test sets with stratified sampling
    :param df: DataFrame to split
    :param target_column: the name of the target/response column
    :param test_size: proportion of the dataset to include in the test split
    :param random_state: controls the shuffling applied to the data before applying the split
    :return: training and test DataFrames
    """
    x = df.drop(target_column, axis=1)
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    return x_train, x_test, y_train, y_test


def preprocessing_features_pipeline():
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    binary_cols = ['Gender', 'family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    multilabel_columns = ['CAEC', 'CALC', 'MTRANS']

    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    binary_categorical_transformer = Pipeline(steps=[
        ('ordinal', OrdinalEncoder())
    ])

    multi_categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('binary', binary_categorical_transformer, binary_cols),
            ('multi', multi_categorical_transformer, multilabel_columns)
        ])
    return preprocessor


class ClassificationPipeline:
    """
    A class to represent a classification pipeline
    """
    __slots__ = ['data_folder', 'file_name', 'x_train', 'x_test', 'y_train', 'y_test', 'params_list', 'pipe_list']

    def __init__(self, data_folder, file_name):
        """
        Constructor for the ClassificationPipeline class
        :param data_folder:
        :param file_name:
        """
        self.data_folder = data_folder
        self.file_name = file_name
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.params_list = None
        self.pipe_list = None
        logging.basicConfig(level=logging.INFO)

    def load_and_preprocess_data(self) -> None:
        """
        Load and preprocess the data
        :return:
        """
        try:
            df = pd.read_csv(f"{self.data_folder}/{self.file_name}")
        except FileNotFoundError as e:
            logging.error(f"File {self.file_name} not found in folder {self.data_folder}.")
            raise e
        self.x_train, self.x_test, self.y_train, self.y_test = stratified_split_train_test(df, "NObeyesdad")
        le = LabelEncoder()
        self.y_train = le.fit_transform(self.y_train)
        self.y_test = le.transform(self.y_test)

    def get_classifier_params_and_pipelines(self) -> None:
        """
        Get the classifier parameters and pipelines
        :return:
        """
        self.params_list = [
            {
                'clf_lr__C': uniform(loc=0, scale=4),
            },
            {
                'clf_knn__n_neighbors': range(1, 31),
                'clf_knn__weights': ['uniform', 'distance'],
                'clf_knn__metric': ['euclidean', 'manhattan']
            },
            {
                'clf_nb__var_smoothing': uniform(loc=0, scale=1)
            },
            {
                'clf_rf__n_estimators': [100, 200, 500],
                'clf_rf__max_features': ['sqrt', 'log2'],
                'clf_rf__max_depth': [4, 5, 6, 7, 8],
                'clf_rf__criterion': ['gini', 'entropy']
            },
            {
                'clf_svm__C': [0.1, 1, 10, 100],
                'clf_svm__gamma': [1, 0.1, 0.01, 0.001],
                'clf_svm__kernel': ['rbf', 'linear', 'poly']
            },
            {
                'clf_lgbm__n_estimators': [100, 200, 500],
                'clf_lgbm__learning_rate': [0.01, 0.1, 1],
                'clf_lgbm__max_depth': [4, 5, 6, 7, 8]
            },
            {
                'clf_nn__hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
                'clf_nn__activation': ['tanh', 'relu'],
                'clf_nn__solver': ['sgd', 'adam'],
                'clf_nn__alpha': [0.0001, 0.05],
                'clf_nn__learning_rate': ['constant', 'adaptive'],
            }
        ]
        self.pipe_list = [
            Pipeline([('preprocessor', preprocessing_features_pipeline()),
                      ('clf_lr', LogisticRegression(random_state=42, max_iter=1000))]),
            Pipeline([('preprocessor', preprocessing_features_pipeline()),
                      ('clf_knn', KNeighborsClassifier())]),
            Pipeline([('preprocessor', preprocessing_features_pipeline()),
                      ('clf_nb', GaussianNB())]),
            Pipeline([('preprocessor', preprocessing_features_pipeline()),
                      ('clf_rf', RandomForestClassifier())]),
            Pipeline([('preprocessor', preprocessing_features_pipeline()),
                      ('clf_svm', SVC())]),
            Pipeline([('preprocessor', preprocessing_features_pipeline()),
                      ('clf_lgbm', LGBMClassifier())]),
            Pipeline([('preprocessor', preprocessing_features_pipeline()),
                      ('clf_nn', MLPClassifier(max_iter=3000))])
        ]

    def fit_and_evaluate_classifier(self, pipe: Pipeline, params: dict) -> None:
        """
        Fit and evaluate a classifier
        :param pipe:
        :param params:
        :return:
        """
        clf = RandomizedSearchCV(pipe, params, random_state=0, scoring='f1_micro', cv=5, n_iter=20, n_jobs=-1)
        search = clf.fit(self.x_train, self.y_train)
        y_pred = search.predict(self.x_test)
        cm = confusion_matrix(self.y_test, y_pred)
        print(cm)

        precision = precision_score(self.y_test, y_pred, average='micro')
        recall = recall_score(self.y_test, y_pred, average='micro')
        f1 = f1_score(self.y_test, y_pred, average='micro')

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

    def fit_and_evaluate_all_classifiers(self) -> None:
        """
        Fit and evaluate all classifiers
        :return:
        """
        for params, pipe in zip(self.params_list, self.pipe_list):
            self.fit_and_evaluate_classifier(pipe, params)

    def run(self) -> None:
        """
        Run the pipeline
        :return:
        """
        self.load_and_preprocess_data()
        self.get_classifier_params_and_pipelines()
        self.fit_and_evaluate_all_classifiers()


def execute_pipeline():
    """
    Execute the pipeline
    :return:
    """
    pipeline = ClassificationPipeline(DATA_FOLDER, FILE_NAME)
    pipeline.run()


if __name__ == "__main__":
    execute_pipeline()
