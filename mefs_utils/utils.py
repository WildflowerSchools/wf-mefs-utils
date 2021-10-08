import pandas as pd

def to_datetime(object):
    try:
        datetime = pd.to_datetime(object, utc=True).to_pydatetime()
        if pd.isnull(datetime):
            date = None
    except:
        datetime = None
    return datetime

def to_date(object):
    try:
        date = pd.to_datetime(object).date()
        if pd.isnull(date):
            date = None
    except:
        date = None
    return date

def to_singleton(object):
    try:
        num_elements = len(object)
        if num_elements > 1:
            raise ValueError('More than one element in object. Conversion to singleton failed')
        if num_elements == 0:
            return None
        return object[0]
    except:
        return object

def to_boolean(object):
    if isinstance(object, bool):
        return object
    if isinstance(object, str):
        if object in ['True', 'TRUE', 'T']:
            return True
        if object in ['False', 'FALSE', 'F']:
            return False
        return None
    if isinstance(object, int):
        if object == 1:
            return True
        if object == 0:
            return False
        return None
    return None

def extract_alphanumeric(object):
    if pd.isna(object):
        return None
    try:
        object_string = str(object)
    except:
        return None
    alphanumeric_string = ''.join(ch for ch in object_string if ch.isalnum())
    return alphanumeric_string

def infer_school_year(
    date,
    rollover_month=7,
    rollover_day=31
):
    if pd.isna(date):
        return None
    if date.month <= rollover_month and date.day <= rollover_day:
        return '{}-{}'.format(
            date.year - 1,
            date.year
        )
    else:
        return '{}-{}'.format(
            date.year,
            date.year + 1
        )

def filter_dataframe(
    dataframe,
    filter_dict=None
):
    if filter_dict is None:
        return dataframe
    index_columns = dataframe.index.names
    dataframe=dataframe.reset_index()
    for key, value_list in filter_dict.items():
        dataframe = dataframe.loc[dataframe[key].isin(value_list)]
    dataframe.set_index(index_columns, inplace=True)
    return dataframe

def select_from_dataframe(
    dataframe,
    select_dict=None
):
    if select_dict is None:
        return dataframe
    keys, values = zip(*select_dict.items())
    dataframe = dataframe.xs(values, level=keys)
    return dataframe
