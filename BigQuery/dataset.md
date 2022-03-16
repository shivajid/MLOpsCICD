## About the Dataset
The sample dataset contains obfuscated Google Analytics 360 data from the Google Merchandise Store, a real ecommerce store. The Google Merchandise Store sells Google branded merchandise. The data is typical of what you would see for an ecommerce website. It includes the following kinds of information:

Traffic source data: information about where website visitors originate. This includes data about organic traffic, paid search traffic, display traffic, etc.
Content data: information about the behavior of users on the site. This includes the URLs of pages that visitors look at, how they interact with content, etc.
Transactional data: information about the transactions that occur on the Google Merchandise Store websi

Because it provides Google Analytics 360 data from an ecommerce website, the dataset is useful for exploring the benefits of exporting Google Analytics 360 data into BigQuery via the integration. Once you have access to the dataset you can run queries such as those in this guide for the period of 1-Aug-2016 to 1-Aug-2017. For example to see the total pageviews the website received for 1-Jan-2017 you would query the dataset with:
`SELECT SUM(totals.pageviews) AS TotalPageviews

FROM 'bigquery-public-data.google_analytics_sample.ga_sessions_20170101'

`

### Limitations
All users have viewer access to the dataset. This means that you can query the dataset and generate reports but you cannot complete administrative tasks. Data for some fields have been obfuscated such as fullVisitorId, or removed such as clientId, adWordsClickInfo and geoNetwork. “Not available in demo dataset” will be the returned for STRING values and “null” will be returned for INTEGER values, when querying the fields that contain no data.

