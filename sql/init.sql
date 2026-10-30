CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS employees (
    stt BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ma_nv VARCHAR(20) NOT NULL UNIQUE,
    ho_va_ten VARCHAR(150) NOT NULL,
    dob DATE,
    gender VARCHAR(10),
    phong_ban VARCHAR(100),
    chuc_vu VARCHAR(100),
    sdt VARCHAR(20),
    email VARCHAR(254) UNIQUE,
    embedding VECTOR(512) NOT NULL
);

--CREATE INDEX ON employees USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);