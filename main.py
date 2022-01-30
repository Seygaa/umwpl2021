import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from seaborn import heatmap
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
import matplotlib.pyplot as plt


def print_results_classification(model_name, accuracy, f1):
    print("\t\t", model_name)
    print("\t\t\t Accuracy: ", accuracy)
    print("\t\t\t F1: ", f1, '\n')


def print_results_regression(model_name, r2, rmse):
    print("\t\t", model_name)
    print("\t\t\t R^2: ", r2)
    print("\t\t\t RMSE: ", rmse, '\n')


def print_results(name, accuracy, f1, r2, rmse):
    print(name, '\n')
    print("\tClassification", '\n')
    print_results_classification("Logistic Regression: ", accuracy[0], f1[0])
    print_results_classification("Random Forest: ", accuracy[1], f1[1])
    print_results_classification("SVM: ", accuracy[2], f1[2])
    print_results_classification("Naive Bayes:", accuracy[3], f1[3])
    print("\tRegression", '\n')
    print_results_regression("Linear regression: ", r2[0], rmse[0])
    print_results_regression("LASSO regression: ", r2[1], rmse[1])
    print_results_regression("Ridge regression: ", r2[2], rmse[2])
    print("\tClassification after Regression", '\n')
    print_results_classification("Linear regression: ", accuracy[4], f1[4])
    print_results_classification("LASSO regression: ", accuracy[5], f1[5])
    print_results_classification("Ridge regression: ", accuracy[6], f1[6])


def logistic_regression(X_train, y_train, X_test):
    logit = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    return logit.predict(X_test)


def random_forest_classifier(X_train, y_train, X_test):
    forest = RandomForestClassifier().fit(X_train, y_train)
    return forest.predict(X_test)


def support_vector_machine(X_train, y_train, X_test):
    svc = SVC().fit(X_train, y_train)
    return svc.predict(X_test)


def naive_bayes(X_train, y_train, X_test):
    bayes = BernoulliNB(binarize=None).fit(X_train, y_train)
    return bayes.predict(X_test)


def linear_regression(X_train, y_train, X_test):
    linear = LinearRegression().fit(X_train, y_train)
    return linear.predict(X_test)


def lasso_regression(X_train, y_train, X_test):
    lasso = Lasso().fit(X_train, y_train)
    return lasso.predict(X_test)


def ridge_regression(X_train, y_train, X_test):
    ridge = Ridge(alpha=4).fit(X_train, y_train)
    return ridge.predict(X_test)


def approximation_ki(y):
    y[y <= 100] = 1
    y[y > 100] = 0


@ignore_warnings(category=ConvergenceWarning)
def models_generator(data):
    X = data[data.columns[1:data.shape[1]]].copy()
    y = data[data.columns[0]].copy()
    approximation_ki(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, stratify=y)

    y_pred_logit = logistic_regression(X_train, y_train, X_test)
    y_pred_forest = random_forest_classifier(X_train, y_train, X_test)
    y_pred_svc = support_vector_machine(X_train, y_train, X_test)
    y_pred_bayes = naive_bayes(X_train, y_train, X_test)

    X_train, X_test, y_train, y_test = train_test_split(X, data[data.columns[0]], test_size=0.2, random_state=1,
                                                        stratify=y)
    y_pred_linear = linear_regression(X_train, y_train, X_test)
    y_pred_lasso = lasso_regression(X_train, y_train, X_test)
    y_pred_ridge = ridge_regression(X_train, y_train, X_test)

    R2 = np.array(
        [np.round(r2_score(y_test, y_pred_linear), decimals=2), np.round(r2_score(y_test, y_pred_lasso), decimals=2),
         np.round(r2_score(y_test, y_pred_ridge), decimals=2)])

    Rmse = np.array(
        [np.round(mean_squared_error(y_test, y_pred_linear, squared=False), decimals=2),
         np.round(mean_squared_error(y_test, y_pred_lasso, squared=False), decimals=2),
         np.round(mean_squared_error(y_test, y_pred_ridge, squared=False), decimals=2)])

    approximation_ki(y_test)
    approximation_ki(y_pred_linear)
    approximation_ki(y_pred_lasso)
    approximation_ki(y_pred_ridge)

    accuracy = np.array([np.round(accuracy_score(y_test, y_pred_logit), decimals=2),
                         np.round(accuracy_score(y_test, y_pred_forest), decimals=2),
                         np.round(accuracy_score(y_test, y_pred_svc), decimals=2),
                         np.round(accuracy_score(y_test, y_pred_bayes), decimals=2),
                         np.round(accuracy_score(y_test, y_pred_linear), decimals=2),
                         np.round(accuracy_score(y_test, y_pred_lasso), decimals=2),
                         np.round(accuracy_score(y_test, y_pred_ridge), decimals=2)])
    F1 = np.array(
        [np.round(f1_score(y_test, y_pred_logit), decimals=2), np.round(f1_score(y_test, y_pred_forest), decimals=2),
         np.round(f1_score(y_test, y_pred_svc), decimals=2), np.round(f1_score(y_test, y_pred_bayes), decimals=2),
         np.round(f1_score(y_test, y_pred_linear), decimals=2), np.round(f1_score(y_test, y_pred_lasso), decimals=2),
         np.round(f1_score(y_test, y_pred_ridge), decimals=2)])

    return accuracy, F1, R2, Rmse


def metrics_generator(df_delta_extfp, df_delta_klekfp, df_delta_maccsfp, df_kappa_extfp, df_kappa_klekfp,
                      df_kappa_maccsfp, df_mu_extfp, df_mu_klekfp, df_mu_maccsfp):
    accuracy_delta_extfp, F1_delta_extfp, R2_delta_extfp, Rmse_delta_extfp = models_generator(df_delta_extfp)
    accuracy_delta_klekfp, F1_delta_klekfp, R2_delta_klekfp, Rmse_delta_klekfp = models_generator(df_delta_klekfp)
    accuracy_delta_maccsfp, F1_delta_maccsfp, R2_delta_maccsfp, Rmse_delta_maccsfp = models_generator(df_delta_maccsfp)

    accuracy_kappa_extfp, F1_kappa_extfp, R2_kappa_extfp, Rmse_kappa_extfp = models_generator(df_kappa_extfp)
    accuracy_kappa_klekfp, F1_kappa_klekfp, R2_kappa_klekfp, Rmse_kappa_klekfp = models_generator(df_kappa_klekfp)
    accuracy_kappa_maccsfp, F1_kappa_maccsfp, R2_kappa_maccsfp, Rmse_kappa_maccsfp = models_generator(df_kappa_maccsfp)

    accuracy_mu_extfp, F1_mu_extfp, R2_mu_extfp, Rmse_mu_extfp = models_generator(df_mu_extfp)
    accuracy_mu_klekfp, F1_mu_klekfp, R2_mu_klekfp, Rmse_mu_klekfp = models_generator(df_mu_klekfp)
    accuracy_mu_maccsfp, F1_mu_maccsfp, R2_mu_maccsfp, Rmse_mu_maccsfp = models_generator(df_mu_maccsfp)

    delta_accuracy = np.array((accuracy_delta_extfp, accuracy_delta_klekfp, accuracy_delta_maccsfp))
    delta_f1 = np.array((F1_delta_extfp, F1_delta_klekfp, F1_delta_maccsfp))
    delta_r2 = np.array((R2_delta_extfp, R2_delta_klekfp, R2_delta_maccsfp))
    delta_rmse = np.array((Rmse_delta_extfp, Rmse_delta_klekfp, Rmse_delta_maccsfp))
    kappa_accuracy = np.array((accuracy_kappa_extfp, accuracy_kappa_klekfp, accuracy_kappa_maccsfp))
    kappa_f1 = np.array((F1_kappa_extfp, F1_kappa_klekfp, F1_kappa_maccsfp))
    kappa_r2 = np.array((R2_kappa_extfp, R2_kappa_klekfp, R2_kappa_maccsfp))
    kappa_rmse = np.array((Rmse_kappa_extfp, Rmse_kappa_klekfp, Rmse_kappa_maccsfp))
    mu_accuracy = np.array((accuracy_mu_extfp, accuracy_mu_klekfp, accuracy_mu_maccsfp))
    mu_f1 = np.array((F1_mu_extfp, F1_mu_klekfp, F1_mu_maccsfp))
    mu_r2 = np.array((R2_mu_extfp, R2_mu_klekfp, R2_mu_maccsfp))
    mu_rmse = np.array((Rmse_mu_extfp, Rmse_mu_klekfp, Rmse_mu_maccsfp))

    print_results("delta_extfp", accuracy_delta_extfp, F1_delta_extfp, R2_delta_extfp, Rmse_delta_extfp)
    print_results("delta_klekfp", accuracy_delta_klekfp, F1_delta_klekfp, R2_delta_klekfp, Rmse_delta_klekfp)
    print_results("delta_maccsfp", accuracy_delta_maccsfp, F1_delta_maccsfp, R2_delta_maccsfp, Rmse_delta_maccsfp)
    print_results("kappa_extfp", accuracy_kappa_extfp, F1_kappa_extfp, R2_kappa_extfp, Rmse_kappa_extfp)
    print_results("kappa_klekfp", accuracy_kappa_klekfp, F1_kappa_klekfp, R2_kappa_klekfp, Rmse_kappa_klekfp)
    print_results("kappa_maccsfp", accuracy_kappa_maccsfp, F1_kappa_maccsfp, R2_kappa_maccsfp, Rmse_kappa_maccsfp)
    print_results("mu_extfp", accuracy_mu_extfp, F1_mu_extfp, R2_mu_extfp, Rmse_mu_extfp)
    print_results("mu_klekfp", accuracy_mu_klekfp, F1_mu_klekfp, R2_mu_klekfp, Rmse_mu_klekfp)
    print_results("mu_maccsfp", accuracy_mu_maccsfp, F1_mu_maccsfp, R2_mu_maccsfp, Rmse_mu_maccsfp)

    return delta_accuracy, delta_f1, delta_r2, delta_rmse, kappa_accuracy, kappa_f1, kappa_r2, kappa_rmse, mu_accuracy, mu_f1, mu_r2, mu_rmse


def regression_heatmap(name, metrics):
    heat_map = heatmap(metrics, annot=True,
                       xticklabels=("linear", "LASSO", "ridge"),
                       yticklabels=("extfp", "klekfp", "maccsfp"))
    figure = heat_map.get_figure()
    figure.savefig('results/{}.png'.format(name), dpi=figure.dpi)
    plt.clf()


def classification_heatmap(name, metrics):
    heat_map = heatmap(metrics, annot=True,
                       xticklabels=("logistic", "random forest", "SVM", "naive bayes", "linear", "LASSO", "ridge"),
                       yticklabels=("extfp", "klekfp", "maccsfp"))
    figure = heat_map.get_figure()
    figure.savefig('results/{}.png'.format(name), dpi=figure.dpi)
    plt.clf()


def heatmap_generator(delta_accuracy, delta_f1, delta_r2, delta_rmse, kappa_accuracy, kappa_f1, kappa_r2,
                      kappa_rmse, mu_accuracy, mu_f1, mu_r2, mu_rmse):
    regression_heatmap("delta_r2", delta_r2)
    regression_heatmap("delta_rmse", delta_rmse)
    regression_heatmap("kappa_r2", kappa_r2)
    regression_heatmap("kappa_rmse", kappa_rmse)
    regression_heatmap("mu_r2", mu_r2)
    regression_heatmap("mu_rmse", mu_rmse)

    classification_heatmap("delta_accuracy", delta_accuracy)
    classification_heatmap("delta_f1", delta_f1)
    classification_heatmap("kappa_accuracy", kappa_accuracy)
    classification_heatmap("kappa_f1", kappa_f1)
    classification_heatmap("mu_accuracy", mu_accuracy)
    classification_heatmap("mu_f1", mu_f1)


def datasets_handle():
    df_delta_extfp = pd.read_csv('datasets/delta_opioid_ExtFP_ready.csv').dropna()
    df_delta_klekfp = pd.read_csv('datasets/delta_opioid_KlekFP_ready.csv').dropna()
    df_delta_maccsfp = pd.read_csv('datasets/delta_opioid_MACCSFP_ready.csv').dropna()
    df_kappa_extfp = pd.read_csv('datasets/kappa_opioid_ExtFP_ready.csv').dropna()
    df_kappa_klekfp = pd.read_csv('datasets/kappa_opioid_KlekFP_ready.csv').dropna()
    df_kappa_maccsfp = pd.read_csv('datasets/kappa_opioid_MACCSFP_ready.csv').dropna()
    df_mu_extfp = pd.read_csv('datasets/mu_opioid_ExtFP_ready.csv').dropna()
    df_mu_klekfp = pd.read_csv('datasets/mu_opioid_KlekFP_ready.csv').dropna()
    df_mu_maccsfp = pd.read_csv('datasets/mu_opioid_MACCSFP_ready.csv').dropna()

    delta_accuracy, delta_f1, delta_r2, delta_rmse, kappa_accuracy, kappa_f1, kappa_r2, kappa_rmse, mu_accuracy, mu_f1, mu_r2, mu_rmse = metrics_generator(
        df_delta_extfp,
        df_delta_klekfp,
        df_delta_maccsfp,
        df_kappa_extfp,
        df_kappa_klekfp,
        df_kappa_maccsfp,
        df_mu_extfp,
        df_mu_klekfp,
        df_mu_maccsfp)

    heatmap_generator(delta_accuracy, delta_f1, delta_r2, delta_rmse, kappa_accuracy, kappa_f1, kappa_r2,
                      kappa_rmse, mu_accuracy, mu_f1, mu_r2, mu_rmse)


if __name__ == '__main__':
    datasets_handle()
