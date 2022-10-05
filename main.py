import pandas
import numpy
from matplotlib import pyplot
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree

def calculate_iv(dataset: pandas.DataFrame, feature: str, target: str):
    if dataset[index].dtype != object:
        if dataset[index].nunique() <= 10:
            dataset[index] = dataset[index].map(str)
        else:
            dataset[index] = pandas.cut(dataset[index], 10)
    lst = []
    for i in range(dataset[feature].nunique()):
        val = list(dataset[feature].unique())[i]
        lst.append({
            'Value': val,
            'All': dataset[dataset[feature] == val].count()[feature],
            'Good': dataset[(dataset[feature] == val) & (dataset[target] == 0)].count()[feature],
            'Bad': dataset[(dataset[feature] == val) & (dataset[target] == 1)].count()[feature]
        })

    dset = pandas.DataFrame(lst)
    dset['Distr_Good'] = dset['Good'] / dset['Good'].sum()
    dset['Distr_Bad'] = dset['Bad'] / dset['Bad'].sum()
    dset['WoE'] = numpy.log(dset['Distr_Good'] / dset['Distr_Bad'])
    dset = dset.replace({'WoE': {numpy.inf: 0, -numpy.inf: 0}})
    dset['IV'] = (dset['Distr_Good'] - dset['Distr_Bad']) * dset['WoE']

    return dset['IV'].sum()


def dummies(df: pandas.DataFrame, var: str, dep: str):
    if df[var].dtype == object:
        df[var] = df[var].str.slice(0, 10)
        df[var] = df[var].str.replace(' ', '_')
        freq1 = df[var].value_counts().to_frame()
        freq1.columns = ['freq']
        freq1['freq%'] = df[var].value_counts() / N
        freq1[var] = freq1.index
        freq1.loc[freq1['freq%'] < 0.04, var] = 'OTHERS'
        df.loc[~df[var].isin(freq1[var].unique().tolist()), var] = 'OTHERS'
    if calculate_iv(dataset=df[[var,dep]], feature=var, target=dep) > 0.01:
        df = df.join(pandas.get_dummies(df[var], prefix=var, prefix_sep="_"))
    df = df.drop(columns=[var])
    return df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    DEBUG_MODE = True
    dep = 'PAR90'
    input_dir = 'C:/Users/Admin/PycharmProjects/pythonProject/'
    data_frame = pandas.read_csv(input_dir + 'Result_1.csv')
    target = data_frame[dep]
    drop_list = ['loan_key', 'company_key', 'loan_id', 'lms_loan_id', 'product_id', 'PARX_ost', 'PAR30_ost',
                 'PARX', 'PAR30', 'main_company_name', 'approved_date', 'disbursed_date', 'maturity_date',
                 'loan_status', 'default_status', 'company_name', 'company_status', 'loan_principal', 'CNT',
                 'tenure_at_org', 'days_past_due', 'principal_paid', 'interest_paid', 'loan_interest',
                 'total_outstanding', 'total_paid', 'toal_gross_disb_ost', 'toal_gross_prin_paid', 'toal_fee_paid',
                 'middle_name', 'last_name', 'account_holder_name', 'first_name', 'origination_fee', 'company_id',
                 'borrower_key', 'permanent_resident_since', 'total_amount','Disb_month']
    data_frame = data_frame.drop(columns=drop_list)
    data_types = data_frame.dtypes.to_frame()
    data_types.columns = ['data_type']
    N = data_frame.shape[0]
    for index, row in data_types.iterrows():
        print(index, row['data_type'])
        if index == 'PAR90':
            continue
        if row['data_type'] in ['float64', 'int64']:
            data_frame[index] = data_frame[index].fillna(0)
            if calculate_iv(dataset=data_frame.copy(), feature=index, target=dep) <= 0.01:
                data_frame = data_frame.drop(columns=[index])
        elif row['data_type'] == object:
            data_frame[index] = data_frame[index].fillna('MISSING')
            data_frame = dummies(df=data_frame.copy(), var=index, dep=dep)
        elif row['data_type'] == bool:
            data_frame[index] = data_frame[index].fillna(False)
            data_frame = dummies(df=data_frame.copy(), var=index, dep=dep)
        else:
            print(row['data_type'])
            exit()
    var_list = data_frame.columns.to_list()
    var_list.remove(dep)
    for x in var_list:
        if calculate_iv(dataset=data_frame, feature=x, target=dep) < 0.01:
            data_frame = data_frame.drop(columns=[x])
    data_frame['target'] = target
    data_frame.to_csv("C:/Users/Admin/PycharmProjects/pythonProject/results_2.csv", index=False, sep=",")
    var_list = data_frame.columns.to_list()
    var_list.remove(dep)
    X = data_frame.loc[:, var_list]
    y = data_frame.loc[:, dep]
    model_tree = DecisionTreeClassifier(min_samples_split=10, max_depth=5, min_samples_leaf=20)
    model_tree.fit(X, y)
    pyplot.figure(figsize=(20, 10))


    plot_tree(model_tree,
           feature_names = var_list, #Feature names
           class_names = ["0","1"], #Class names
           rounded = True,
           filled = True)

    pyplot.show()
