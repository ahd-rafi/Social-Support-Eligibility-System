-- Initialize Langfuse database
-- This script creates the langfuse database if it doesn't exist

SELECT 'CREATE DATABASE langfuse'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'langfuse')\gexec
