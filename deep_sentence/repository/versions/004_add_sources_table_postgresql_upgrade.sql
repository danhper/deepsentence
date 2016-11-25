CREATE TABLE sources (
  id          serial PRIMARY KEY,
  url         varchar(255) NOT NULL,
  title       varchar(255),
  article_id  integer NOT NULL REFERENCES articles (id),
  content     text,
  posted_at   timestamp
);

CREATE INDEX ON sources (article_id);
