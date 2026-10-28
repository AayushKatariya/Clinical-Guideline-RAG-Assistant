-- The text-to-SQL tool connects as this role. Even if the SQL guard were bypassed,
-- the database itself refuses writes. Change the password outside local development.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'clinrag_readonly') THEN
        CREATE ROLE clinrag_readonly LOGIN PASSWORD 'readonly';
    END IF;
END
$$;

GRANT CONNECT ON DATABASE clinical TO clinrag_readonly;
GRANT USAGE ON SCHEMA public TO clinrag_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO clinrag_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO clinrag_readonly;
ALTER ROLE clinrag_readonly SET statement_timeout = '5s';
ALTER ROLE clinrag_readonly SET default_transaction_read_only = on;
