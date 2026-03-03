-- Set the default search_path for the database to include ag_catalog and paradedb
-- This ensures that the 'cypher' function and other utilities are available globally
ALTER DATABASE postgres SET search_path = public, ag_catalog, paradedb;

-- Also set it for the current session to ensure the migration finishes with the right context
SET search_path = public, ag_catalog, paradedb;
