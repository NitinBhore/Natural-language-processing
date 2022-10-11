"""Import library"""
import pandas_gbq

"""
Blueprint of BigQuery operations like load the data on biqQuery table and query the table
Args:
    project_id = Project ID
    table_id = Dataset with table name
    dataframe = Dataframe Name
    sql_query = Sql query
"""
class BigQueryOps():  # pylint: disable=too-few-public-methods
    """Blueprint of BigQuery operations like load the data on biqQuery table and query the table"""
    def __init__(self, project_id):
        """Inits the big querry operations"""
        self.project_id = project_id

    def load_data(self, table_id, dataframe):
        """Load a DataFrame to BigQuery with pandas-gbq"""
        self.table_id = table_id
        self.dataframe = dataframe
        pandas_gbq.to_gbq(self.dataframe, self.table_id, project_id=self.project_id)

    def query_table(self, sql_query):
        """Run a query with pandas-gbq"""
        self.sql_query = sql_query
        dataframe = pandas_gbq.read_gbq(self.sql_query, project_id=self.project_id)
        return dataframe
