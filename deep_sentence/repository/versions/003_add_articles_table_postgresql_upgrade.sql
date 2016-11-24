CREATE TABLE articles (
  id          serial PRIMARY KEY,
  remote_id   varchar(40) NOT NULL UNIQUE,
  title       varchar(255) NOT NULL UNIQUE,
  url         varchar(255) NOT NULL UNIQUE,
  service_id  integer NOT NULL REFERENCES services (id),
  category_id integer NOT NULL REFERENCES categories (id),
  content     text,
  posted_at   timestamp
);

CREATE INDEX ON articles (service_id);
CREATE INDEX ON articles (category_id);
