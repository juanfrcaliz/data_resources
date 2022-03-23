import pandas as pd
from sqlalchemy import create_engine
import configparser


def get_db_credentials(file: str = 'db_access.cfg') -> dict:
    """
    This function receives a filename as a string, corresponding with
    a config file with the credentials of the PADRE database.
    The function looks for the file and extracts the credentials.
    Returns the credentials inside a dictionary with the keys:
    - user
    - password
    - host
    - port
    - db
    """
    config = configparser.ConfigParser()
    config.read(file)

    cred_dict = {}
    credentials_ = ['user', 'password', 'host', 'port', 'db']

    for cred in credentials_:
        val = config.get('credentials', cred)
        cred_dict[cred] = val

    return cred_dict


def extraction_query(query_: str, credentials_: dict) -> pd.DataFrame:
    """
    This function receives an SQL query as a string and the database
    credentials stored in a dictionary.
    The query must be written, so it returns a table, which will be
    returned by the function as a Pandas DataFrame.
    """
    user = credentials_['user']
    password = credentials_['password']
    host = credentials_['host']
    port = credentials_['port']
    db = credentials_['db']

    engine = f'mysql+pymysql://{user}:{password}@{host}:{port}/{db}'
    sqlengine = create_engine(engine)

    with sqlengine.connect() as dbConnection:
        applications_ = pd.read_sql(query_, dbConnection)

    return applications_
  
