import wf_core_data.utils
import pandas as pd
import numpy as np
import inflection
import collections
import itertools
import copy
import os
import logging

logger = logging.getLogger(__name__)

TIME_FRAME_ID_VARIABLES = [
    'school_year'
]

STUDENT_ID_VARIABLES = [
    'rs_id'
]

STUDENT_INFO_VARIABLES = [
    'child_id_mefs',
    'first_name',
    'last_name',
    'birth_month_year',
    'gender',
    'ethnicity'
]

STUDENT_ASSIGNMENT_VARIABLES = [
    'group_name_mefs'
]

RESULTS_VARIABLES = [
    'test_date',
    'total_score',
    'standard_score',
    'standard_score_category',
    'percentile'
]

DEFAULT_MIN_GROWTH_DAYS = 60

DEFAULT_SCHOOL_YEAR_DURATION_MONTHS = 9

DEFAULT_ROLLOVER_MONTH = 7
DEFAULT_ROLLOVER_DAY = 31

def fetch_results_local_directory(
    path,
    file_extensions=['.csv', '.CSV']
):
    if not os.path.exists(path):
        raise ValueError('Path \'{}\' not found'.format(path))
    if not os.path.isdir(path):
        raise ValueError('Object at \'{}\' is not a directory'.format(path))
    paths = list()
    for directory_entry in os.listdir(path):
        file_path = os.path.join(
            path,
            directory_entry
        )
        if not os.path.isfile(file_path):
            continue
        file_extension = os.path.splitext(os.path.normpath(file_path))[1]
        if file_extension not in file_extensions:
            continue
        paths.append(file_path)
    if len(paths) == 0:
        raise ValueError('No files of type {} found in directory'.format(file_extensions))
    results = fetch_results_local_files(paths)
    return results

def fetch_results_local_files(
    paths
):
    results_list = list()
    for path in paths:
        results_file = fetch_results_local_file(
            path=path
        )
        results_list.append(results_file)
    results = pd.concat(results_list)
    return results

def fetch_results_local_file(
    path,
    rollover_month=DEFAULT_ROLLOVER_MONTH,
    rollover_day=DEFAULT_ROLLOVER_DAY
):
    if not os.path.exists(path):
        raise ValueError('File \'{}\' not found'.format(path))
    if not os.path.isfile(path):
        raise ValueError('Object at \'{}\' is not a file'.format(path))
    results = pd.read_csv(
        path,
        dtype='object'
    )
    results['test_date'] = results['Date of Test'].apply(wf_core_data.utils.to_date)
    results['school_year'] = results['test_date'].apply(lambda x: wf_core_data.utils.infer_school_year(
        wf_core_data.utils.to_date(x),
        rollover_month=rollover_month,
        rollover_day=rollover_day
    ))
    return results

def parse_results(results):
    test_events = extract_test_events(results)
    student_info, student_info_changes = extract_student_info(results)
    student_assignments = extract_student_assignments(results)
    return test_events, student_info, student_info_changes, student_assignments

def extract_test_events(
    results
):
    test_events = (
        results
        .rename(columns={
            'RS ID': 'rs_id',
            'Percentile (National)': 'percentile',
            'Total Score': 'total_score',
            'Standard Score': 'standard_score',
            'Standard Score Category': 'standard_score_category'
        })
    )
    test_events['percentile'] = pd.to_numeric(test_events['percentile']).astype('float')
    test_events['total_score'] = pd.to_numeric(test_events['total_score']).astype('float')
    test_events['standard_score'] = pd.to_numeric(test_events['standard_score']).astype('float')
    test_events = test_events.reindex(columns=list(itertools.chain(
        TIME_FRAME_ID_VARIABLES,
        STUDENT_ID_VARIABLES,
        RESULTS_VARIABLES
    )))
    test_events.set_index(
        list(itertools.chain(
            TIME_FRAME_ID_VARIABLES,
            STUDENT_ID_VARIABLES
        )),
        inplace=True
    )
    test_events.sort_index(inplace=True)
    return test_events

def extract_student_info(
    results
):
    student_info = (
        results
        .rename(columns= {
            'RS ID': 'rs_id',
            'Child ID': 'child_id_mefs',
            'First Name': 'first_name',
            'Last Name': 'last_name',
            'Birth Month / Year': 'birth_month_year',
            'Gender': 'gender',
            'Ethnicity': 'ethnicity'
        })
    )
    student_info = (
        student_info
        .reindex(columns=list(itertools.chain(
            STUDENT_ID_VARIABLES,
            TIME_FRAME_ID_VARIABLES,
            STUDENT_INFO_VARIABLES,
            ['test_date']
        )))
        .drop_duplicates()
    )
    student_info_changes = (
        student_info
        .groupby(STUDENT_ID_VARIABLES)
        .filter(lambda group: len(group.drop_duplicates(subset=STUDENT_INFO_VARIABLES)) > 1)
    )
    student_info = (
        student_info
        .sort_values('test_date')
        .drop(columns='test_date')
        .drop(columns=TIME_FRAME_ID_VARIABLES)
        .groupby(STUDENT_ID_VARIABLES)
        .tail(1)
        .set_index(STUDENT_ID_VARIABLES)
        .sort_index()
    )
    return student_info, student_info_changes

def extract_student_assignments(
    results
):
    student_assignments = (
        results
        .rename(columns= {
            'RS ID': 'rs_id',
            'Group Name': 'group_name_mefs'
        })
    )
    student_assignments = (
        student_assignments
        .reindex(columns=list(itertools.chain(
            STUDENT_ID_VARIABLES,
            TIME_FRAME_ID_VARIABLES,
            STUDENT_ASSIGNMENT_VARIABLES
        )))
        .drop_duplicates()
        .set_index(list(itertools.chain(
            STUDENT_ID_VARIABLES,
            TIME_FRAME_ID_VARIABLES
        )))
        .sort_index()
    )
    return student_assignments

def summarize_by_test(
    test_events,
    student_assignments,
    grouping_variables = [
        'school_year',
        'group_name_mefs',
    ],
    filter_dict=None,
    select_dict=None
):
    tests = (
        test_events
        .join(
            student_assignments,
            how='left',
            on=[
                'rs_id',
                'school_year'
            ]
        )
        .groupby(grouping_variables)
        .agg(
            num_test_events=('test_date', 'count'),
            num_valid_total_score=('total_score', 'count'),
            num_valid_standard_score=('standard_score', 'count'),
            num_valid_percentile=('percentile', 'count')
        )
    )
    tests = tests.loc[tests['num_test_events'] > 0].copy()
    if filter_dict is not None:
        tests = wf_core_data.utils.filter_dataframe(
            dataframe=tests,
            filter_dict=filter_dict
        )
    if select_dict is not None:
        tests = wf_core_data.utils.select_from_dataframe(
            dataframe=tests,
            select_dict=select_dict
        )
    return tests

def summarize_by_student(
    test_events,
    student_info,
    student_assignments,
    new_time_index=['school_year'],
    min_growth_days=DEFAULT_MIN_GROWTH_DAYS,
    school_year_duration_months=DEFAULT_SCHOOL_YEAR_DURATION_MONTHS,
    filter_dict=None,
    select_dict=None
):
    new_index_variables = list(itertools.chain(
        new_time_index,
        STUDENT_ID_VARIABLES
    ))
    total_scores = (
        test_events
        .dropna(subset=['total_score'])
        .sort_values('test_date')
        .groupby(new_index_variables)
        .agg(
            total_score_starting_date=('test_date', lambda x: x.dropna().iloc[0]),
            total_score_ending_date=('test_date', lambda x: x.dropna().iloc[-1]),
            starting_total_score=('total_score', lambda x: x.dropna().iloc[0]),
            ending_total_score=('total_score', lambda x: x.dropna().iloc[-1]),
        )
    )
    percentiles = (
        test_events
        .dropna(subset=['percentile'])
        .sort_values('test_date')
        .groupby(new_index_variables)
        .agg(
            percentile_starting_date=('test_date', lambda x: x.dropna().iloc[0]),
            percentile_ending_date=('test_date', lambda x: x.dropna().iloc[-1]),
            starting_percentile=('percentile', lambda x: x.dropna().iloc[0]),
            ending_percentile=('percentile', lambda x: x.dropna().iloc[-1]),
        )
    )
    students = total_scores.join(
        percentiles,
        how='outer'
    )
    students['total_score_num_days'] = (
        np.subtract(
            students['total_score_ending_date'],
            students['total_score_starting_date']
        )
        .apply(lambda x: x.days)
    )
    students['total_score_growth'] = np.subtract(
        students['ending_total_score'],
        students['starting_total_score']
    )
    students.loc[students['total_score_num_days'] < min_growth_days, 'total_score_growth'] = np.nan
    students['total_score_growth_per_school_year'] = 365.25*(school_year_duration_months/12)*students['total_score_growth']/students['total_score_num_days']
    students['percentile_num_days'] = (
        np.subtract(
            students['percentile_ending_date'],
            students['percentile_starting_date']
        )
        .apply(lambda x: x.days)
    )
    students['percentile_growth'] = np.subtract(
        students['ending_percentile'],
        students['starting_percentile']
    )
    students.loc[students['percentile_num_days'] < min_growth_days, 'percentile_growth'] = np.nan
    students['percentile_growth_per_school_year'] = 365.25*(school_year_duration_months/12)*students['percentile_growth']/students['percentile_num_days']
    students = students.join(
        student_info,
        how='left',
        on=['rs_id']
    )
    students['met_goal'] = students['ending_percentile'].apply(
        lambda x: (x >= 50) if not pd.isna(x) else None
    )
    latest_student_assignments = (
        student_assignments
        .reset_index()
        .sort_values(['school_year'])
        .groupby(list(itertools.chain(
            STUDENT_ID_VARIABLES,
            new_time_index
        )))
        .tail(1)
        .set_index(list(itertools.chain(
            STUDENT_ID_VARIABLES,
            new_time_index
        )))
    )
    students = students.join(
        latest_student_assignments,
        how='left',
        on=latest_student_assignments.index.names
    )
    students = students.reindex(columns=list(itertools.chain(
        STUDENT_INFO_VARIABLES,
        STUDENT_ASSIGNMENT_VARIABLES,
        [
            'total_score_starting_date',
            'total_score_ending_date',
            'total_score_num_days',
            'starting_total_score',
            'ending_total_score',
            'total_score_growth',
            'total_score_growth_per_school_year',
            'percentile_starting_date',
            'percentile_ending_date',
            'percentile_num_days',
            'starting_percentile',
            'ending_percentile',
            'percentile_growth',
            'percentile_growth_per_school_year',
            'met_goal'
        ]
    )))
    if filter_dict is not None:
        students = wf_core_data.utils.filter_dataframe(
            dataframe=students,
            filter_dict=filter_dict
        )
    if select_dict is not None:
        students = wf_core_data.utils.select_from_dataframe(
            dataframe=students,
            select_dict=select_dict
        )
    return students

def summarize_by_group(
    students,
    grouping_variables=[
        'school_year',
        'group_name_mefs'
    ],
    filter_dict=None,
    select_dict=None
):
    if len(grouping_variables) == 0:
        grouping_variables = lambda x: 0
    groups = (
        students
        .reset_index()
        .groupby(grouping_variables)
        .agg(
            num_test_results=('rs_id', 'count'),
            num_valid_starting_total_score=('starting_total_score', 'count'),
            mean_starting_total_score=('starting_total_score', 'mean'),
            starting_total_score_sd=('starting_total_score', 'std'),
            num_valid_ending_total_score=('ending_total_score', 'count'),
            mean_ending_total_score=('ending_total_score', 'mean'),
            ending_total_score_sd=('ending_total_score', 'std'),
            num_valid_total_score_growth=('total_score_growth', 'count'),
            mean_total_score_growth=('total_score_growth', 'mean'),
            total_score_growth_sd=('total_score_growth', 'std'),
            mean_total_score_growth_per_school_year=('total_score_growth_per_school_year', 'mean'),
            total_score_growth_per_school_year_sd=('total_score_growth_per_school_year', 'std'),
            num_valid_starting_percentile=('starting_percentile', 'count'),
            mean_starting_percentile=('starting_percentile', 'mean'),
            starting_percentile_sd=('starting_percentile', 'std'),
            num_valid_ending_percentile=('ending_percentile', 'count'),
            mean_ending_percentile=('ending_percentile', 'mean'),
            ending_percentile_sd=('ending_percentile', 'std'),
            num_valid_percentile_growth=('percentile_growth', 'count'),
            mean_percentile_growth=('percentile_growth', 'mean'),
            percentile_growth_sd=('percentile_growth', 'std'),
            mean_percentile_growth_per_school_year=('percentile_growth_per_school_year', 'mean'),
            percentile_growth_per_school_year_sd=('percentile_growth_per_school_year', 'std'),
            num_valid_goal_info=('ending_percentile', 'count'),
            num_met_goal=('met_goal', 'sum')
        )
        .dropna(how='all')
    )
    groups = groups.loc[groups['num_test_results'] > 0].copy()
    groups['frac_met_goal'] = groups['num_met_goal'].astype('float')/groups['num_valid_goal_info'].astype('float')
    groups['mean_starting_total_score_sem'] = np.divide(
        groups['starting_total_score_sd'],
        np.sqrt(groups['num_valid_starting_total_score'])
    )
    groups['mean_ending_total_score_sem'] = np.divide(
        groups['ending_total_score_sd'],
        np.sqrt(groups['num_valid_ending_total_score'])
    )
    groups['mean_total_score_growth_sem'] = np.divide(
        groups['total_score_growth_sd'],
        np.sqrt(groups['num_valid_total_score_growth'])
    )
    groups['mean_total_score_growth_per_school_year_sem'] = np.divide(
        groups['total_score_growth_per_school_year_sd'],
        np.sqrt(groups['num_valid_total_score_growth'])
    )
    groups['mean_starting_percentile_sem'] = np.divide(
        groups['starting_percentile_sd'],
        np.sqrt(groups['num_valid_starting_percentile'])
    )
    groups['mean_ending_percentile_sem'] = np.divide(
        groups['ending_percentile_sd'],
        np.sqrt(groups['num_valid_ending_percentile'])
    )
    groups['mean_percentile_growth_sem'] = np.divide(
        groups['percentile_growth_sd'],
        np.sqrt(groups['num_valid_percentile_growth'])
    )
    groups['mean_percentile_growth_per_school_year_sem'] = np.divide(
        groups['percentile_growth_per_school_year_sd'],
        np.sqrt(groups['num_valid_percentile_growth'])
    )
    groups = groups.reindex(columns=[
        'num_test_results',
        'num_valid_starting_total_score',
        'mean_starting_total_score',
        'starting_total_score_sd',
        'mean_starting_total_score_sem',
        'num_valid_ending_total_score',
        'mean_ending_total_score',
        'ending_total_score_sd',
        'mean_ending_total_score_sem',
        'num_valid_total_score_growth',
        'mean_total_score_growth',
        'total_score_growth_sd',
        'mean_total_score_growth_sem',
        'mean_total_score_growth_per_school_year',
        'total_score_growth_per_school_year_sd',
        'mean_total_score_growth_per_school_year_sem',
        'num_valid_starting_percentile',
        'mean_starting_percentile',
        'starting_percentile_sd',
        'mean_starting_percentile_sem',
        'num_valid_ending_percentile',
        'mean_ending_percentile',
        'ending_percentile_sd',
        'mean_ending_percentile_sem',
        'num_valid_percentile_growth',
        'mean_percentile_growth',
        'percentile_growth_sd',
        'mean_percentile_growth_sem',
        'mean_percentile_growth_per_school_year',
        'percentile_growth_per_school_year_sd',
        'mean_percentile_growth_per_school_year_sem',
        'num_valid_goal_info',
        'frac_met_goal'
    ])
    if filter_dict is not None:
        groups = wf_core_data.utils.filter_dataframe(
            dataframe=groups,
            filter_dict=filter_dict
        )
    if select_dict is not None:
        groups = wf_core_data.utils.select_from_dataframe(
            dataframe=groups,
            select_dict=select_dict
        )
    return groups
