# Data Formats

## ASearcher collections

Source: `/home/juyiang/code/Agentic-RAG-R1/data/ASearcher` (see `ASearcher` folder copied from the Agentic-RAG-R1 project).

All parquet files inside this directory share the same two-column schema:

| column    | dtype | description |
|-----------|-------|-------------|
| `question` | string | Natural-language question that may require multi-hop retrieval. |
| `answer`   | string | Gold answer span or entity in free text. |

### Available splits

| file name | rows | notes |
|-----------|------|-------|
| `train.parquet` | 35,583 | Mix of open-domain questions, intended for supervised data generation. |
| `test_2wikimultihopqa_rand1000.parquet` | 1,000 | Random 2WikiMultiHopQA subset. |
| `test_bamboogle.parquet` | 125 | Difficult factual QA questions from BambooGL. |
| `test_frames.parquet` | 824 | Frame reasoning benchmark. |
| `test_gaia.parquet` | 103 | GAIA temporal/spatial reasoning tasks. |
| `test_hotpotqa_rand1000.parquet` | 1,000 | Random HotpotQA subset. |
| `test_musique_rand1000.parquet` | 1,000 | MuSiQue subset focusing on multi-hop questions. |
| `test_nq_rand1000.parquet` | 1,000 | Natural Questions sample. |
| `test_popqa_rand1000.parquet` | 1,000 | PopQA open-domain sample. |
| `test_triviaqa_rand1000.parquet` | 1,000 | TriviaQA random sample. |
| `test_xbench_deepsearch.parquet` | 100 | Chinese DeepSearch eval set.

(Counts recorded via `pandas.read_parquet` during inspection.)

### Example rows

```
{"question": "What nationality is the director of film Queensland (Film)?", "answer": "Australian"}
{"question": "If my future wife has the same first name as the 15th first lady ... what is my future wife's name?", "answer": "Jane Ballou"}
{"question": "截至2024年12月31日...“最高价”与“最低价”之差约为多少元/克？", "answer": "161.27元"}
```

These samples illustrate that every record provides a plain question plus a short canonical answer, which can be fed directly into the data generation pipeline built in this repository.
