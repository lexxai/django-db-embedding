-- Create Ukrainian text search dictionary using Hunspell
CREATE TEXT SEARCH DICTIONARY ukrainian_hunspell (
    TEMPLATE = ispell,
    DictFile = uk_UA,
    AffFile = uk_UA,
    StopWords = ukrainian
);

-- Create Ukrainian text search configuration
CREATE TEXT SEARCH CONFIGURATION ukrainian ( COPY = simple );

ALTER TEXT SEARCH CONFIGURATION ukrainian
    ALTER MAPPING FOR word, asciiword, hword, hword_part
    WITH ukrainian_hunspell, simple;


