import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import shap
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, median_absolute_error, \
    mean_absolute_percentage_error
from sklearn.model_selection import KFold, train_test_split
import warnings
warnings.filterwarnings("ignore")

def calculate_metrics(y_true, y_pred):
    metrics = {
        '均方误差[MSE]': mean_squared_error(y_true, y_pred),
        '均方根误差[RMSE]': np.sqrt(mean_squared_error(y_true, y_pred)),
        '平均绝对误差[MAE]': mean_absolute_error(y_true, y_pred),
        '平均绝对百分比误差[MAPE]': mean_absolute_percentage_error(y_true, y_pred),
        '决定系数[R²]': r2_score(y_true, y_pred)
    }
    return metrics


class BPNNModel:
    def __init__(self, data, x, y, test_num=0.2, solver='lbfgs', learning_rate_init=0.001,
                 learning_rate="constant", alpha=0.0001, max_iter=200):
        self.data = data
        self.x = x
        self.y = y
        self.test_num = test_num
        self.solver = solver
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.max_iter = max_iter
        self.calculate_metrics = calculate_metrics
        self.model = None

    def bp_neural_network(self):
        test_num = round(self.test_num, 1)
        if (0 <= (1 - test_num) * len(self.data) < 1) or (0 <= test_num * len(self.data) < 1):
            test_num = 1 / len(self.data)

        xy1 = self.x + self.y
        data = self.data[xy1]
        xy2 = ['X' + str(i + 1) for i in range(len(self.x))] + ['Y']
        var_explain = pd.DataFrame([f"{x}：{y}" for x, y in zip(xy2, xy1)], columns=['变量解释'])
        x = xy2[:-1]
        y = xy2[-1]
        data.columns = xy2

        # 数据准备
        X = data[x].values
        y = data[y].values.ravel()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_num, random_state=42)

        # 特征标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 初始化神经网络
        self.model = MLPRegressor(solver=self.solver, learning_rate_init=self.learning_rate_init,
                                  learning_rate=self.learning_rate, alpha=self.alpha,
                                  max_iter=self.max_iter, random_state=42)

        # 训练模型
        self.model.fit(X_train, y_train)

        # 预测
        y_train_pred = self.model.predict(X_train)
        y_pred = self.model.predict(X_test)

        results_df = pd.DataFrame({
            '真实值': y_test,
            '预测值': y_pred
        })
        # 使用SHAP进行模型解释
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)

        # SHAP值可视化
        shap.summary_plot(shap_values, X_test, feature_names=self.x, plot_type="bar")

        # 模型评估
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_pred)

        # 将评估指标转换为DataFrame
        results = pd.DataFrame({
            '训练集误差': train_metrics.values(),
            '测试集误差': test_metrics.values()
        }, index=train_metrics.keys())

        # 特征重要性
        from sklearn.inspection import permutation_importance

        result = permutation_importance(self.model, X_test, y_test, n_repeats=30, random_state=42)
        importance = result.importances_mean
        feature_importance_df = pd.DataFrame({'特征': self.x, '重要性': importance})
        # 筛选特征，设定阈值
        threshold = 0.00001  # 可以根据需要调整
        selected_features = feature_importance_df[feature_importance_df['重要性'] > threshold]

        return var_explain, results, results_df, selected_features

class GBDTModel:
    def __init__(self, data, x, y, test_num=0.2, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.data = data
        self.x = x
        self.y = y
        self.test_num = test_num
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.calculate_metrics = calculate_metrics
        self.model = None

    def train(self):
        # 数据准备
        xy = self.x + self.y
        data = self.data[xy]

        data.columns = ['X' + str(i + 1) for i in range(len(self.x))] + ['Y']

        X = data.iloc[:, :-1].values  # 输入特征
        y = data.iloc[:, -1].values.ravel()  # 目标变量，转换为一维数组

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_num, random_state=42)

        # 特征标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 初始化梯度提升树回归模型
        self.model = GradientBoostingRegressor(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                                max_depth=self.max_depth, random_state=42)

        # 训练模型
        self.model.fit(X_train, y_train)

        # 预测
        y_train_pred = self.model.predict(X_train)
        y_pred = self.model.predict(X_test)

        results_df = pd.DataFrame({
            '真实值': y_test,
            '预测值': y_pred
        })

        # 模型评估
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_pred)

        # 将评估指标转换为DataFrame
        results = pd.DataFrame({
            '训练集误差': train_metrics.values(),
            '测试集误差': test_metrics.values()
        }, index=train_metrics.keys())

        # 返回参数列表和预测结果
        params = pd.DataFrame({
            '指标名称': [
                '模型',
                '树的个数',
                '学习率',
                '最大深度',
                '均方根误差'
            ],
            '具体值': [
                '梯度提升树回归',
                self.n_estimators,
                self.learning_rate,
                self.max_depth,
                f'{test_metrics["均方根误差[RMSE]"]:.4f}'
            ]
        })
        # 特征重要性
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({'特征': self.x, '重要性': importances})

        # 筛选特征，设定阈值
        threshold = 0.00001  # 可以根据需要调整
        selected_features = feature_importance_df[feature_importance_df['重要性'] > threshold]

        # 使用SHAP进行模型解释
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)

        # SHAP值可视化
        shap.summary_plot(shap_values, X_test, feature_names=self.x, plot_type="bar")

        return params, results, results_df, selected_features

class RandomForestModel:
    def __init__(self, data, x, y, test_num=0.2, n_estimators=100, max_depth=None):
        self.data = data
        self.x = x
        self.y = y
        self.test_num = test_num
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.calculate_metrics = calculate_metrics
        self.model = None

    def train(self):
        # 数据准备
        xy = self.x + self.y
        data = self.data[xy]

        data.columns = ['X' + str(i + 1) for i in range(len(self.x))] + ['Y']

        X = data.iloc[:, :-1].values  # 输入特征
        y = data.iloc[:, -1].values.ravel()  # 目标变量，转换为一维数组

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_num, random_state=42)

        # 特征标准化
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 初始化随机森林回归模型
        self.model = RandomForestRegressor(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42)

        # 训练模型
        self.model.fit(X_train, y_train)

        # 预测
        y_train_pred = self.model.predict(X_train)
        y_pred = self.model.predict(X_test)

        results_df = pd.DataFrame({
            '真实值': y_test,
            '预测值': y_pred
        })

        # 模型评估
        train_metrics = self.calculate_metrics(y_train, y_train_pred)
        test_metrics = self.calculate_metrics(y_test, y_pred)

        # 将评估指标转换为DataFrame
        results = pd.DataFrame({
            '训练集误差': train_metrics.values(),
            '测试集误差': test_metrics.values()
        }, index=train_metrics.keys())

        # 特征重要性
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({'特征': self.x, '重要性': importances})

        # 筛选特征，设定阈值
        threshold = 0.00001  # 可以根据需要调整
        selected_features = feature_importance_df[feature_importance_df['重要性'] > threshold]

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test)

        # SHAP值可视化
        shap.summary_plot(shap_values, X_test, feature_names=self.x, plot_type="bar")

        # 返回参数列表、评估结果和特征重要性
        params = pd.DataFrame({
            '指标名称': [
                '模型',
                '树的个数',
                '最大深度',
                '均方根误差'
            ],
            '具体值': [
                '随机森林回归',
                self.n_estimators,
                self.max_depth,
                f'{test_metrics["均方根误差[RMSE]"]:.4f}'
            ]
        })

        return params, results, results_df, selected_features


class XGBoostModel:
    def __init__(self, data: pd.DataFrame, x: list, y: list, test_num=0.2,
                 cv_folds: int = 5, use_bayesian_optimization=False):
        self.data = data
        self.x = x
        self.y = y
        self.test_num = test_num
        self.cv_folds = cv_folds
        self.use_bayesian_optimization = use_bayesian_optimization


    def regression_metrics(self, true: np.ndarray, pred: np.ndarray):
        MSE = mean_squared_error(true, pred)
        RMSE = np.sqrt(MSE)
        MAE = mean_absolute_error(true, pred)
        MedianAE = median_absolute_error(true, pred)
        MAPE = mean_absolute_percentage_error(true, pred)
        R2 = r2_score(true, pred)

        result = pd.DataFrame(index=['均方误差[MSE]', '均方根误差[RMSE]', '平均绝对误差[MAE]',
                                     '绝对误差中位数[MedianAE]', '平均绝对百分比误差[MAPE]', '决定系数'],
                              data=[MSE, RMSE, MAE, MedianAE, MAPE, R2])

        return result

    def train(self):
        test_num = round(self.test_num, 1)
        if (0 <= (1 - test_num) * len(self.data) < 1) or (0 <= test_num * len(self.data) < 1):
            test_num = 1 / len(self.data)

        xy1 = self.x + self.y
        data = self.data[xy1]

        xy2 = ['X' + str(i + 1) for i in range(len(self.x))] + ['Y']
        var_explain = pd.DataFrame([f"{x}：{y}" for x, y in zip(xy2, xy1)], columns=['变量解释'])
        data.columns = xy2

        x_train, x_test, y_train, y_test = train_test_split(data[xy2[:-1]], data[xy2[-1]],
                                                            test_size=test_num, random_state=2023)

        params = {
            'n_estimators': 1000,
            'learning_rate': 0.08,
            "subsample": 0.75,
            "colsample_bytree": 1,
            "max_depth": 7,
            "gamma": 0
        }

        params_df = pd.DataFrame([params]).T
        params_df.reset_index(inplace=True)
        params_df.columns = ['模型参数', '具体值']

        # 交叉验证
        kf = KFold(n_splits=self.cv_folds, shuffle=True, random_state=2023)
        cv_results = []

        for train_index, val_index in kf.split(x_train):
            X_train_fold, X_val_fold = x_train.iloc[train_index], x_train.iloc[val_index]
            y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

            xgb_model = XGBRegressor(**params)
            xgb_model.fit(X_train_fold, y_train_fold)

            # 评估模型
            val_pred = xgb_model.predict(X_val_fold)
            cv_results.append(mean_squared_error(y_val_fold, val_pred))

        # 打印交叉验证结果
        cv_results = pd.DataFrame(np.array(cv_results).reshape(1, -1),
                                  columns=[f'{i}折' for i in range(1, self.cv_folds + 1)])
        mean_column = pd.DataFrame([[np.mean(cv_results)]], columns=['平均值'])
        cv_results = pd.concat([cv_results, mean_column], axis=1)
        cv_results = cv_results.T
        cv_results.columns = ['均方误差']

        if self.use_bayesian_optimization:
            param_space = {
                'n_estimators': (100, 2000),
                'learning_rate': (0.01, 0.3, 'uniform'),
                'max_depth': (3, 10),
                'subsample': (0.1, 1.0, 'uniform'),
                'colsample_bytree': (0.1, 1.0, 'uniform'),
            }
            from skopt import BayesSearchCV
            opt = BayesSearchCV(
                XGBRegressor(),
                param_space,
                n_iter=200,
                cv=self.cv_folds,
                n_jobs=-1,
                random_state=2023
            )

            opt.fit(x_train, y_train)
            xgb_model = opt.best_estimator_
            xgb_params = opt.best_params_

        else:
            xgb_model = XGBRegressor(**params)  # params 是默认参数
            xgb_model.fit(x_train, y_train)

        # 训练集预测和评估
        pred_train = xgb_model.predict(x_train)
        train_metrics = self.regression_metrics(y_train, pred_train)

        # 测试集预测和评估
        pred_test = xgb_model.predict(x_test)
        results_df = pd.DataFrame({
            '真实值': y_test,
            '预测值': pred_test
        }).reset_index(drop=False).rename(columns={'index': '样本索引'})

        # 特征重要性
        importances = xgb_model.feature_importances_
        feature_importance_df = pd.DataFrame({'特征': self.x, '重要性': importances})

        # 筛选特征，设定阈值
        threshold = 0.00001  # 可以根据需要调整
        selected_features = feature_importance_df[feature_importance_df['重要性'] > threshold]

        metric = self.regression_metrics(y_test, pred_test)

        # 使用SHAP进行模型解释
        # 创建 SHAP 解释器
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(x_test)
        feature_importance = np.abs(shap_values).mean(axis=0)
        feature_names = x_test.columns  # 如果 x_test 是 DataFrame
        importance_df = pd.DataFrame(list(zip(feature_names, feature_importance)), columns=["Feature", "Importance"])
        importance_df = importance_df.sort_values(by="Importance", ascending=False)

        print(importance_df)

        # 总体影响可视化
        # shap.summary_plot(shap_values, x_test, feature_names=self.x)

        # 对单个样本进行详细分析（比如第一个样本）
        # shap.initjs()  # 只在 Jupyter Notebook 中需要
        # plt.figure(figsize=(10, 8))
        # shap.force_plot(explainer.expected_value, shap_values[0], x_test.iloc[0], feature_names=self.x, matplotlib=True)
        #
        plt.show()
        return var_explain, params_df, cv_results, results_df, selected_features, metric, train_metrics


