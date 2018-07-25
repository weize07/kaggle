X = pd.read_csv(os.path.join(DIR,'files/unzipped_data/application_train.csv'), nrows=None)

X['CODE_GENDER'].replace('XNA',np.nan, inplace=True)
X['CODE_GENDER'].value_counts()
X['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
X['car_to_birth_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_BIRTH']
X['car_to_employ_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_EMPLOYED']
X['children_ratio'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
X['credit_to_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
X['credit_to_goods_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
X['credit_to_income_ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
X['income_per_child'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
X['phone_to_birth_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH']
X['phone_to_employ_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_EMPLOYED']

# External sources
X['external_sources_weighted'] = X.EXT_SOURCE_1 * 2 + X.EXT_SOURCE_2 * 3 + X.EXT_SOURCE_3 * 4
for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
    X['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
        X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)

engineered_numerical_columns = ['annuity_income_percentage',
                                'car_to_birth_ratio',
                                'car_to_employ_ratio',
                                'children_ratio',
                                'credit_to_annuity_ratio',
                                'credit_to_goods_ratio',
                                'credit_to_income_ratio',
                                'days_employed_percentage',
                                'income_credit_percentage',
                                'income_per_child',
                                'income_per_person',
                                'payment_rate',
                                'phone_to_birth_ratio',
                                'phone_to_employ_ratio',
                                'external_sources_weighted',
                                'external_sources_min',
                                'external_sources_max',
                                'external_sources_sum',
                                'external_sources_mean',
                                'external_sources_nanmedian']

# ### Aggregation features
AGGREGATION_RECIPIES = [
    (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [('AMT_ANNUITY', 'max'),
                                              ('AMT_CREDIT', 'max'),
                                              ('EXT_SOURCE_1', 'mean'),
                                              ('EXT_SOURCE_2', 'mean'),
                                              ('OWN_CAR_AGE', 'max'),
                                              ('OWN_CAR_AGE', 'sum')]),
    (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                            ('AMT_INCOME_TOTAL', 'mean'),
                                            ('DAYS_REGISTRATION', 'mean'),
                                            ('EXT_SOURCE_1', 'mean')]),
    (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                 ('CNT_CHILDREN', 'mean'),
                                                 ('DAYS_ID_PUBLISH', 'mean')]),
    (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                           ('EXT_SOURCE_2', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                  ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                  ('APARTMENTS_AVG', 'mean'),
                                                  ('BASEMENTAREA_AVG', 'mean'),
                                                  ('EXT_SOURCE_1', 'mean'),
                                                  ('EXT_SOURCE_2', 'mean'),
                                                  ('EXT_SOURCE_3', 'mean'),
                                                  ('NONLIVINGAREA_AVG', 'mean'),
                                                  ('OWN_CAR_AGE', 'mean'),
                                                  ('YEARS_BUILD_AVG', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                            ('EXT_SOURCE_1', 'mean')]),
    (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                           ('CNT_CHILDREN', 'mean'),
                           ('CNT_FAM_MEMBERS', 'mean'),
                           ('DAYS_BIRTH', 'mean'),
                           ('DAYS_EMPLOYED', 'mean'),
                           ('DAYS_ID_PUBLISH', 'mean'),
                           ('DAYS_REGISTRATION', 'mean'),
                           ('EXT_SOURCE_1', 'mean'),
                           ('EXT_SOURCE_2', 'mean'),
                           ('EXT_SOURCE_3', 'mean')]),
]

groupby_aggregate_names = []
for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
    group_object = X.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        X = X.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)

diff_feature_names = []
for groupby_cols, specs in tqdm(AGGREGATION_RECIPIES):
    for select, agg in tqdm(specs):
        if agg in ['mean','median','max','min']:
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            diff_name = '{}_diff'.format(groupby_aggregate_name)
            abs_diff_name = '{}_abs_diff'.format(groupby_aggregate_name)

            X[diff_name] = X[select] - X[groupby_aggregate_name]
            X[abs_diff_name] = np.abs(X[select] - X[groupby_aggregate_name])

            diff_feature_names.append(diff_name)
            diff_feature_names.append(abs_diff_name)

X['DAYS_EMPLOYED'].replace(365243,np.nan, inplace=True)
X['long_employment'] = (X['DAYS_EMPLOYED'] < -2000).astype(int)
X['retirement_age'] = (X['DAYS_BIRTH'] < -14000).astype(int)
X['cnt_non_child'] = X['CNT_FAM_MEMBERS'] - X['CNT_CHILDREN']
X['child_to_non_child_ratio'] = X['CNT_CHILDREN'] / X['cnt_non_child']
X['income_per_non_child'] = X['AMT_INCOME_TOTAL'] / X['cnt_non_child']
X['credit_per_person'] = X['AMT_CREDIT'] / X['CNT_FAM_MEMBERS']
X['credit_per_child'] = X['AMT_CREDIT'] / (1 + X['CNT_CHILDREN'])
X['credit_per_non_child'] = X['AMT_CREDIT'] / X['cnt_non_child']
application = X

# -----------bureau
bureau = pd.read_csv('./data/bureau.csv')
bureau['bureau_credit_active_binary'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)
bureau['bureau_credit_enddate_binary'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
groupby_SK_ID_CURR = bureau.groupby(by=['SK_ID_CURR'])

features = pd.DataFrame({'SK_ID_CURR':bureau['SK_ID_CURR'].unique()})

group_object = groupby_SK_ID_CURR['DAYS_CREDIT'].agg('count').reset_index()
group_object.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

group_object = groupby_SK_ID_CURR['CREDIT_TYPE'].agg('nunique').reset_index()
group_object.rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

features['bureau_average_of_past_loans_per_type'] = features['bureau_number_of_past_loans'] / features['bureau_number_of_loan_types']

group_object = groupby_SK_ID_CURR['bureau_credit_active_binary'].agg('mean').reset_index()
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

features['bureau_debt_credit_ratio'] = features['bureau_total_customer_debt'] / features['bureau_total_customer_credit']

group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

features['bureau_overdue_debt_ratio'] =     features['bureau_total_customer_overdue'] / features['bureau_total_customer_debt']

group_object = groupby_SK_ID_CURR['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
group_object.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bureau_average_creditdays_prolonged'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

group_object = groupby_SK_ID_CURR['bureau_credit_enddate_binary'].agg('mean').reset_index()
group_object.rename(index=str, columns={'bureau_credit_enddate_binary': 'bureau_credit_enddate_percentage'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

application = application.merge(features,
                                left_on=['SK_ID_CURR'],
                                right_on=['SK_ID_CURR'],
                                how='left',
                                validate='one_to_one')

# ## Aggregations
BUREAU_AGGREGATION_RECIPIES = [('CREDIT_TYPE', 'count'),
                               ('CREDIT_ACTIVE', 'size')
                               ]
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_ANNUITY',
                   'AMT_CREDIT_SUM',
                   'AMT_CREDIT_SUM_DEBT',
                   'AMT_CREDIT_SUM_LIMIT',
                   'AMT_CREDIT_SUM_OVERDUE',
                   'AMT_CREDIT_MAX_OVERDUE',
                   'CNT_CREDIT_PROLONG',
                   'CREDIT_DAY_OVERDUE',
                   'DAYS_CREDIT',
                   'DAYS_CREDIT_ENDDATE',
                   'DAYS_CREDIT_UPDATE'
                   ]:
        BUREAU_AGGREGATION_RECIPIES.append((select, agg))
BUREAU_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], BUREAU_AGGREGATION_RECIPIES)]
groupby_aggregate_names = []
for groupby_cols, specs in tqdm(BUREAU_AGGREGATION_RECIPIES):
    group_object = bureau.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        application = application.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)

#--------------credit_card
credit_card = pd.read_csv(os.path.join(DIR, './data/credit_card_balance.csv'))

# # Preprocessing
credit_card['AMT_DRAWINGS_ATM_CURRENT'][credit_card['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
credit_card['AMT_DRAWINGS_CURRENT'][credit_card['AMT_DRAWINGS_CURRENT'] < 0] = np.nan


# ### Hand crafted features
credit_card['number_of_instalments'] = credit_card.groupby(
    by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()[
    'CNT_INSTALMENT_MATURE_CUM']

credit_card['credit_card_max_loading_of_credit_limit'] = credit_card.groupby(
    by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
    lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]

features = pd.DataFrame({'SK_ID_CURR':credit_card['SK_ID_CURR'].unique()})

group_object = credit_card.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].agg('nunique').reset_index()
group_object.rename(index=str, columns={'SK_ID_PREV': 'credit_card_number_of_loans'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

# It is worth exploring `credit_card_number_of_loans>1` binary version of this variable
group_object= credit_card.groupby(by=['SK_ID_CURR'])['number_of_instalments'].sum().reset_index()
group_object.rename(index=str, columns={'number_of_instalments': 'credit_card_total_instalments'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

features['credit_card_installments_per_loan'] = (
    features['credit_card_total_instalments'] / features['credit_card_number_of_loans'])

group_object = credit_card.groupby(by=['SK_ID_CURR'])['credit_card_max_loading_of_credit_limit'].agg('mean').reset_index()
group_object.rename(index=str, columns={'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

group_object = credit_card.groupby(
    by=['SK_ID_CURR'])['SK_DPD'].agg('mean').reset_index()
group_object.rename(index=str, columns={'SK_DPD': 'credit_card_average_of_days_past_due'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

group_object = credit_card.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

group_object = credit_card.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

features['credit_card_cash_card_ratio'] = features['credit_card_drawings_atm'] / features['credit_card_drawings_total']
application = application.merge(features,
                                left_on=['SK_ID_CURR'],
                                right_on=['SK_ID_CURR'],
                                how='left',
                                validate='one_to_one')

# ## Aggregations
CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_BALANCE',
                   'AMT_CREDIT_LIMIT_ACTUAL',
                   'AMT_DRAWINGS_ATM_CURRENT',
                   'AMT_DRAWINGS_CURRENT',
                   'AMT_DRAWINGS_OTHER_CURRENT',
                   'AMT_DRAWINGS_POS_CURRENT',
                   'AMT_PAYMENT_CURRENT',
                   'CNT_DRAWINGS_ATM_CURRENT',
                   'CNT_DRAWINGS_CURRENT',
                   'CNT_DRAWINGS_OTHER_CURRENT',
                   'CNT_INSTALMENT_MATURE_CUM',
                   'MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES)]
groupby_aggregate_names = []
for groupby_cols, specs in tqdm(CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES):
    group_object = credit_card.groupby(groupby_cols)
    for select, agg in tqdm(specs):
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        application = application.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)
