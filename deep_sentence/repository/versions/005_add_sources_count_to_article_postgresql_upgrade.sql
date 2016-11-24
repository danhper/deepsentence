ALTER TABLE articles ADD sources_count INTEGER NOT NULL DEFAULT 0;
CREATE INDEX ON articles (sources_count);
