CREATE TABLE categories (
  id          serial PRIMARY KEY,
  name        varchar(40) NOT NULL UNIQUE,
  label       varchar(255),
  service_id  integer NOT NULL REFERENCES services (id),
  remote_id   varchar(40) UNIQUE
);

CREATE INDEX ON categories (service_id);
