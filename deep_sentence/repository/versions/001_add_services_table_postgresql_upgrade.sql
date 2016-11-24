CREATE TABLE services (
  id          serial PRIMARY KEY,
  name        varchar(40) NOT NULL UNIQUE
);
