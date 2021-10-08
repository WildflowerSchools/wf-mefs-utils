import wf_core_data.rosters.shared_constants
import pandas as pd
import uuid
import os
import logging

logger = logging.getLogger(__name__)

MEFS_TARGET_COLUMN_NAMES = [
    'FirstName',
    'LastName',
    'ChildID',
    'BirthMonthYear',
    'Gender',
    'SpecialEducationServices',
    'Ethnicity',
    'SecondLanguageLearner',
    'PostalCode',
    'CountryCode',
    'Notes',
    'Class'
]

MEFS_GENDER_MAP = {
    'M': 'Male',
    'F': 'Female',
    'unmatched_value': 'Other',
    'na_value': 'Other'
}

MEFS_ETHNICITY_MAP = {
    'african_american': 'Black / African',
    'asian_american': 'Asian',
    'hispanic': 'Hispanic / Latino',
    'middle_eastern': None,
    'native_american': 'Native / Aboriginal',
    'other': None,
    'pacific_islander': 'Hawaiian / Pacific Islander',
    'white': 'White / Caucasian',
    'unmatched_value': None,
    'na_value': None
}

MEFS_GRADE_NAME_MAP = {
    'EC': 'EC',
    'PK': 'PK',
    'PK_3': 'PK',
    'PK_4': 'PK',
    'K': 'K',
    '1': '1',
    '2': '2',
    '3': '3',
    '4': '4',
    '5': '5',
    '6': '6',
    '7': '7',
    '8': '8',
    '9': '9',
    '10': '10',
    '11': '11',
    '12': '12',
    'unmatched_value': None,
    'na_value': None
}

MEFS_TESTABLE_GRADES = [
    'K',
    '1',
    '2',
    '3',
    '4',
    '5',
    '6',
    '7',
    '8',
    '9',
    '10',
    '11',
    '12'
]

def create_mefs_roster_and_write_locally(
    base_directory,
    filename_suffix,
    master_roster_subdirectory='master_rosters',
    master_roster_filename_stem='master_roster',
    mefs_roster_subdirectory='mefs_rosters',
    mefs_roster_filename_stem='mefs_roster',
    mefs_ids_filename_suffix=None,
    mefs_ids_subdirectory='mefs_ids',
    mefs_ids_filename_stem='mefs_ids'
):
    if mefs_ids_filename_suffix is None:
        mefs_ids_filename_suffix=filename_suffix
    master_roster_filename = os.path.join(
        base_directory,
        master_roster_subdirectory,
        '{}_{}'.format(
            master_roster_filename_stem,
            filename_suffix
        ),
        '{}_{}.pkl'.format(
            master_roster_filename_stem,
            filename_suffix
        )
    )
    master_roster_data = pd.read_pickle(master_roster_filename)
    mefs_ids_filename = os.path.join(
        base_directory,
        mefs_ids_subdirectory,
        '{}_{}'.format(
            mefs_ids_filename_stem,
            mefs_ids_filename_suffix
        ),
        '{}_{}.pkl'.format(
            mefs_ids_filename_stem,
            mefs_ids_filename_suffix
        )
    )
    old_mefs_ids = pd.read_pickle(mefs_ids_filename)
    mefs_roster_data, new_mefs_ids = wf_core_data.create_mefs_roster(
        master_roster_data=master_roster_data,
        mefs_ids=old_mefs_ids
    )
    write_mefs_rosters_local(
        mefs_roster_data=mefs_roster_data,
        base_directory=base_directory,
        subdirectory=mefs_roster_subdirectory,
        filename_stem=mefs_roster_filename_stem,
        filename_suffix=filename_suffix
    )
    write_mefs_ids_local(
        mefs_ids=new_mefs_ids,
        base_directory=base_directory,
        subdirectory=mefs_ids_subdirectory,
        filename_stem=mefs_ids_filename_stem,
        filename_suffix=filename_suffix
    )


def create_mefs_roster(
    master_roster_data,
    mefs_ids
):
    ## Rename fields
    logger.info('Renaming fields')
    mefs_roster_data = (
        master_roster_data
        .rename(columns = {
            'student_first_name_tc': 'FirstName',
            'student_last_name_tc': 'LastName',
            'school_zip_code_tc': 'PostalCode'
        })
    )
    ## Create new fields
    ### Child ID
    logger.info('Creating child ID field')
    logger.info('Current MEFS ID list contains {} IDs'.format(
        len(mefs_ids)
    ))
    mefs_id_map = (
        mefs_ids
        .copy()
        .reset_index()
        .dropna(subset=['school_id_tc', 'student_id_tc'])
        .set_index(['school_id_tc', 'student_id_tc'])
    )
    logger.info('Of these {} MEFS IDs, {} IDs correspond to Transparent Classroom IDs'.format(
        len(mefs_ids),
        len(mefs_id_map)

    ))
    new_tc_ids = mefs_roster_data.index.difference(mefs_id_map.index)
    num_additional_mefs_ids = len(new_tc_ids)
    logger.info('Master roster contains {} Transparent Classroom IDs not in current MEFS IDs'.format(
        num_additional_mefs_ids
    ))
    logger.info('Generating new MEFS IDs')
    while True:
        additional_mefs_id_map = pd.DataFrame(
            {'student_id_mefs_wf': [str(uuid.uuid4())[-10:] for _ in range(num_additional_mefs_ids)]},
            index=new_tc_ids
        )
        new_mefs_id_map = pd.concat((
            mefs_id_map,
            additional_mefs_id_map
        ))
        if not new_mefs_id_map['student_id_mefs_wf'].duplicated().any():
            break
        logger.info('New MEFS ID list contains {} duplicates. Regenerating.'.format(
            new_mefs_id_map['student_id_mefs_wf'].duplicated().sum()
        ))
    mefs_roster_data = (
        mefs_roster_data
        .join(
            new_mefs_id_map,
            how='left'
        )
        .rename(columns={'student_id_mefs_wf': 'ChildID'})
    )
    additional_mefs_ids = (
        additional_mefs_id_map
        .reset_index()
        .set_index('student_id_mefs_wf')
    )
    new_mefs_ids = pd.concat((
        mefs_ids,
        additional_mefs_ids
    ))
    if len(new_mefs_ids) - len(mefs_ids) != num_additional_mefs_ids:
        raise ValueError('{} MEFS child IDs were provided and {} new IDs were generated but new table contains {} IDs'.format(
            len(mefs_ids),
            num_additional_mefs_ids,
            len(new_mefs_ids)
        ))
    logger.info('Provided MEFS child ID table contained {} IDs. {} new IDs were generated. New MEFS child ID table contains {} IDs'.format(
        len(mefs_ids),
        num_additional_mefs_ids,
        len(new_mefs_ids)
    ))
    ### Student birth date
    logger.info('Creating birth month and year field')
    mefs_roster_data['BirthMonthYear'] = mefs_roster_data['student_birth_date_tc'].apply(
        lambda x: x.strftime('%Y-%m-%d')
    )
    ### Student gender
    logger.info('Creating gender field')
    mefs_roster_data['Gender'] = mefs_roster_data['student_gender_wf'].apply(
        lambda x: MEFS_GENDER_MAP.get(x, MEFS_GENDER_MAP.get('unmatched_value')) if pd.notna(x) else MEFS_GENDER_MAP.get('na_value')
    )
    ### Special education services
    logger.info('Creating special education services field')
    mefs_roster_data['SpecialEducationServices'] = 'Unknown'
    ### Student ethnicity
    logger.info('Creating ethnicity field')
    def student_ethnicity_mefs(ethnicity_list):
        if not isinstance(ethnicity_list, list):
            return ''
        ethnicity_list_mefs = list()
        for ethnicity in ethnicity_list:
            ethnicity_mefs = MEFS_ETHNICITY_MAP.get(ethnicity)
            if ethnicity_mefs is not None:
                ethnicity_list_mefs.append(ethnicity_mefs)
        ethnicity_string_mefs = '|'.join(sorted(list(set(ethnicity_list_mefs))))
        return ethnicity_string_mefs
    mefs_roster_data['Ethnicity'] = mefs_roster_data['student_ethnicity_wf'].apply(student_ethnicity_mefs)
    ### Second language learner
    logger.info('Creating second language learner field')
    mefs_roster_data['SecondLanguageLearner'] = 'Unknown'
    ### Country code
    logger.info('Creating country code field')
    mefs_roster_data['CountryCode'] = 'US'
    ## Arrange columns and rows
    logger.info('Rearranging columns and rows')
    mefs_roster_data = (
        mefs_roster_data
        .reindex(columns=(
            wf_core_data.rosters.shared_constants.GROUPING_COLUMN_NAMES +
            MEFS_TARGET_COLUMN_NAMES
        ))
        .sort_values(
            wf_core_data.rosters.shared_constants.GROUPING_COLUMN_NAMES +
            ['FirstName', 'LastName']
        )
    )
    ## Create output
    mefs_roster_data = (
        mefs_roster_data
        .reset_index(drop=True)
        .astype('object')
    )
    return mefs_roster_data, new_mefs_ids

def write_mefs_rosters_local(
    mefs_roster_data,
    base_directory,
    subdirectory='mefs_rosters',
    filename_stem='mefs_roster',
    filename_suffix=None

):
    wf_core_data.rosters.shared_functions.write_rosters_local(
        roster_data=mefs_roster_data,
        base_directory=base_directory,
        subdirectory=subdirectory,
        filename_stem=filename_stem,
        filename_suffix=filename_suffix

    )

def write_mefs_ids_local(
    mefs_ids,
    base_directory,
    subdirectory='mefs_ids',
    filename_stem='mefs_ids',
    filename_suffix=None

):
    if filename_suffix is None:
        filename_suffix = datetime.datetime.now(tz=datetime.timezone.utc).strftime('%Y%m%d')
    logger.info('Filename suffix is {}'.format(filename_suffix))
    output_directory = os.path.join(
        base_directory,
        subdirectory,
        '{}_{}'.format(
            filename_stem,
            filename_suffix
        )
    )
    os.makedirs(output_directory, exist_ok=True)
    filename = '{}_{}'.format(
        filename_stem,
        filename_suffix
    )
    mefs_ids.to_csv(
        os.path.join(
            output_directory,
            '{}.csv'.format(
                filename
            )
        )
    )
    mefs_ids.to_pickle(
        os.path.join(
            output_directory,
            '{}.pkl'.format(
                filename
            )
        )
    )
