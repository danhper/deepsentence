CREATE TABLE medias (
  id            serial PRIMARY KEY,
  base_url      varchar(255) NOT NULL UNIQUE,
  sources_count integer NOT NULL DEFAULT 0
);

ALTER TABLE sources ADD COLUMN media_id INTEGER REFERENCES medias (id);
CREATE INDEX ON sources (media_id);
CREATE INDEX ON sources (sources_count);
