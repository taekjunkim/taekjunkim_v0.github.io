---
title: "Data repositories, Data Pipelines"
excerpt: "RDBMS, NoSQL, Extraction-Transformation-Loading process"
categories:
  - DataScience 
author_profile: false
sidebar:
    title: Data Engineering
    nav: sidebar_DE
usemathjax: true
---


## RDBMS: Relational Database Management Systems

- Relational database is a collection of data organized into a table structure, where the tables can be linked, or related, based on data common to each
- Relational databases use structured query language (SQL) for querying data
- Ideal for the optimized storage, retrieval, and processing of data for large volumes of data
- Each table has a unique set of rows and columns
- Relationships can be defined between tables
- Fields can be restricted to specific data types and values
- Can retrieve millions of records in seconds using SQL for querying data
- Security architecture of relational databases provides greater access control and governance

## Advantages of Relational Database

- Create meaningful information by joining tables
- Flexibility to make changes while the database is in use
- Minimize data redundancy by allowing relationships to be defined between tables
- Offer export and import options that provide ease of backup and disaster recovery
- Are ACID compliant, ensuring accuracy and reliability in database transactions

## Limitations of RDBMS

- Does not work well with semi-structured and unstructured data
- Migration between two RDBMS's is possible only when the source and destination tables have identical schemas and data types
- Entering a value greater than the defined length of a data field results in loss of information

## NoSQL Database

- non-relational database design that provides flexible schemas for the storage and retrieval of data
- Gained greater popularity due to the emergence of cloud computing, big data, and high-volume web and mobile applications
- Chosen for their attributes around scale, performance, and ease of use
- Built for specific data models
- Has flexible schemas that allow programmers to create and manage modern applications
- Do not use a traditional row/column/table database design with fixed schemas
- Do not, typically, use the structured query language to query data

## Four different types of NoSQL databases

- Key-value store
  - Data in key-value database is stored as a collection of key-value pairs
  - A key represents an attribute of the data and is a unique identifier
  - Both keys and values can be anything from simple integers or strings to complex JSON documents
  - Great for storing user session data, user preferences, real-time recommendations, targeted advertising, in-memory data caching
  - Not a great fit if you want to
    - Query data on specific data value
    - Need relationship between data values
    - Need multiple unique keys
- Document based
  - Store each record and its associated data within a single document
  - They enable flexible indexing, powerful ad hoc queries, and analytics over collections of documents
  - Preferred for eCommerce platforms, medical records storage, CRM platforms, and analytics platforms
  - Not a great fit if you want to
    - Run complex search queries
    - Perform multi-operation transactions
- Column based
  - Data is stored in cells grouped as columns of data instead of rows
  - A logical grouping of columns is referred to as a column family
  - All cells corresponding to a column are saved as a continuous disk entry, making access and search easier and faster
  - Great for systems that require heavy write requests, storing time-series data, weather data, and IoT data
  - Not a great fit if you want to
    - Run complex queries
    - Change querying patterns frequently
- Graph based
  - Use a graphical model to represent and store data
  - Useful for visualizing, analyzing, and finding connections between different pieces of data
  - An excellent choice for working with connected data
  - Not a great fit if you want to
    - Process high volumes of transactions

## Advantages of NoSQL

- ability to handle large volumes of structured, semi-structured, and unstructured data
- ability to run as a distributed system scaled across multiple data centers
- an efficient and cost-effective scale-out architecture that provides additional capacity and performance with the addition of new nodes
- Simpler design, better control over availability, and improved scalability that makes it agile, flexible, and support quick iterations

## Extract, Transform, and Load process

- Automated process which includes
  - Gathering raw data
  - Extracting information needed for reporting and analysis
  - Cleaning, standardizing, and transforming data into usable format
  - Loading data into a data repository
- Extraction can be through
  - Batch processing: large chunk of data moved from source to destination at scheduled intervals
  - Stream processing: data pulled in real-time from source, transformed in transit, and loaded into data repository
- Transforming data
  - Standardizing data formats and units of measurement
  - Removing duplicate data
  - Filtering out data that is not required
  - Enriching data
  - Establishing key relationships across tables
  - Applying business rules and data validations
- Loading is the transportation of the processed data in to a data repository
  - Initial loading: populating all of the data in the repository
  - Incremental loading: applying updates and modifications periodically
  - Full refresh: erasing a data table and reloading fresh data
  - Load verification includes checks for
    - Missing or null values
    - Server performance
    - Load failures

## Extract, Load, and Transform process

- Help process large sets of unstructured and non-relational data
- Advantages
  - Shortens the cycle between extraction and delivery
  - Allows you to ingest volumes of raw data as immediately as the data becomes available
  - Affords greater flexibility to analysts and data scientists for exploratory data analytics
  - Transforms only that data which is required for a particular analysis so it can be leveraged for multiple use cases
  - Is more suited to work with Big Data

## Data pipelines

- Encompasses the entire journey of moving data from one system to another, including the ETL process
- Can be used for both batch and streaming data
- Supports both long-running batch queries and smaller interactive queries
- Typically loading a data into a data lake but can also load data into a variety of target destinations including other applications and visualization tools
