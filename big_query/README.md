# BiqQuery Operations

This document describes loading the data on the BiqQuery table and querying the table. Use the pandas-gbq package to load a DataFrame to BigQuery and run a simple query on the BigQuery table.

Args:

    * project_id = Project ID
    * table_id = Dataset with table name
    * dataframe = Dataframe Name
    * sql_query = SQL query

Returns:

    * Dataframe: Returns the executed query result
