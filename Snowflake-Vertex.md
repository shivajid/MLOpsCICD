## Snowflake

Snowflake’s Data Cloud is powered by an advanced data platform provided as Software-as-a-Service (SaaS). Snowflake enables data storage, processing, and analytic solutions that are faster, easier to use, and far more flexible than traditional offerings.

The Snowflake data platform is not built on any existing database technology or “big data” software platforms such as Hadoop. Instead, Snowflake combines a completely new SQL query engine with an innovative architecture natively designed for the cloud. To the user, Snowflake provides all of the functionality of an enterprise analytic database, along with many additional special features and unique capabilities.

## Connector Options

Snowflake provides the following connectors to convert SQL Cursors to Pandas Dataframe.

    pip install "snowflake-connector-python[pandas]"

Use the following methods to read data:
    fetch_pandas_all()
    fetch_pandas_batches()

The details are in the following docs

https://docs.snowflake.com/en/user-guide/python-connector-pandas.html

Limitations: This works well for medium size datatsets

For very large datasets, export the data to  GCS/BigQuery and use it to train data.
