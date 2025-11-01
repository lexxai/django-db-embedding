ARG DB_VERSION=${DB_VERSION:-18}

FROM pgvector/pgvector:pg${DB_VERSION}

# Install contrib, Ispell + Hunspell dictionaries for Ukrainian FTS
ARG DB_VERSION
RUN apt-get update && \
    apt-get install -y \
        "postgresql-contrib-${DB_VERSION}" \
        hunspell-uk && \
    rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/postgresql/${DB_VERSION}/tsearch_data

COPY ./ukrainian_fts.sql /docker-entrypoint-initdb.d/
COPY ./ukrainian.stop /usr/share/postgresql/${DB_VERSION}/tsearch_data
