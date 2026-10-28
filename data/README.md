# Data

Nothing in this folder is committed except this README, the evaluation files in `eval/`
and the `.gitkeep` placeholders. Everything else is produced or supplied locally.

```
data/
├── raw/                 # 1. guideline corpus (you provide)
│   ├── *.pdf | *.html | *.md | *.txt | *.jsonl
│   └── metadata.csv     #    optional, per-file metadata
├── metadata/            #    written by `clinrag metadata` (titles for untitled docs)
│   ├── generated_titles.jsonl   # one record per document (append-only, resumable)
│   ├── titles_review.csv        # written by `clinrag metadata report`
│   └── guidelines.orig.jsonl    # untouched corpus, kept by `clinrag metadata apply`
├── processed/           #    written by `clinrag ingest`
│   ├── documents.jsonl
│   └── chunks.jsonl
├── indexes/             #    written by `clinrag index` (BM25 pickle, FAISS index)
├── structured/          # 2. CSVs for Postgres (you provide, or `clinrag db extract`)
│   ├── recommendations.csv
│   ├── drug_dosing.csv
│   └── diagnostic_thresholds.csv
└── eval/                # 3. evaluation set (committed, except the draft)
    ├── eval_set.example.jsonl   # format reference
    ├── eval_set.draft.jsonl     # written by `clinrag eval build-testset` (not committed)
    ├── eval_set.jsonl           # the reviewed 100-question set used by the harness
    ├── eval_set.reworded.jsonl  # same set, in-corpus questions reworded clinician-style
    └── pool_judgments.jsonl     # pooled relevance judgments behind the retrieval labels
```

## 1. Guideline corpus → `data/raw/`

Either:

- **Hugging Face (quickest):** `python scripts/download_guidelines.py --n 1200` pulls a
  stratified sample of open-access guidelines (WHO, NICE, CDC, CMA, CCO, ICRC, SPOR) from
  `epfl-llm/guidelines` into `data/raw/guidelines.jsonl`; or
- **Your own files:** PDFs / HTML / markdown / text. Scanned PDFs need OCR first.

JSONL rows need `text` (or `clean_text`) and ideally `doc_id`, `title`, `organization`,
`year`, `specialty`, `url`.

`metadata.csv` (optional, for file-based corpora):

```csv
filename,doc_id,title,organization,year,specialty,url
nice/ng136.pdf,nice-ng136,Hypertension in adults: diagnosis and management,NICE,2019,cardiology,https://www.nice.org.uk/guidance/ng136
```

Keep only corpus files in `data/raw/`: every `*.jsonl`, `*.txt`, `*.md` etc. there is ingested.

### Untitled documents (`epfl-llm/guidelines`)

In the open-access dump only NICE documents have titles and URLs. Most WHO, CDC, CMA,
SPOR, CCO and ICRC documents have neither (the field holds the string `"None"`), their
cleaned text has lost its title page, and no document has a publication year. Without a
title, citations read "None > Recommendations" and chunks lose their contextual header.

`clinrag metadata` has the LLM write a **descriptive** title for each untitled document
from its section headings and opening text, 20 documents per call (about 48 calls for
the 946 untitled documents):

```bash
clinrag metadata titles --dry-run      # show the first prompt and the number of calls
clinrag metadata titles --limit 20     # pilot: one call
clinrag metadata report                # counts per publisher + data/metadata/titles_review.csv
clinrag metadata titles                # the rest (resumable; --retry redoes rejected ones)
clinrag metadata apply                 # accepted titles -> data/raw/guidelines.jsonl
```

These are not official titles. They are marked `title_source: generated:<model>`, and a
title is accepted only if at least 60% of its topic words occur in the document. Rejected
or empty ones are listed in the review CSV with the reason. `apply` always starts from
the untouched original (kept as `data/metadata/guidelines.orig.jsonl`), turns `"None"`
placeholders into nulls and gives documents still without a title a plain one such as
`WHO guideline de26c293`. URLs and years stay empty. Afterwards re-run `clinrag ingest`
and `clinrag index`.

## 2. Structured data → Postgres

`clinrag db load` fills the `guidelines` table straight from corpus metadata. The other
three tables come from CSVs whose headers match the column names in
[`sql/001_schema.sql`](../sql/001_schema.sql); `guideline_id` must equal a corpus `doc_id`.

To bootstrap them, `clinrag db extract` asks the LLM to pull doses, thresholds and graded
recommendations out of candidate chunks into `data/structured/*.csv`. Review those files
(each dosing row keeps its source chunk id in `notes`) before loading.

## 3. Evaluation set → `data/eval/eval_set.jsonl`

100 questions, one JSON object per line - see `eval_set.example.jsonl` for every field.

| category       | expected  | n  | what it tests                                              |
|----------------|-----------|----|------------------------------------------------------------|
| `in_corpus`    | answer    | 60 | answerable from guideline text; has `relevant_doc_ids`     |
| `sql`          | answer    | 8  | answerable from the structured tables                      |
| `adversarial`  | answer    | 8  | false premise - must be corrected from the evidence        |
| `adversarial`  | refuse    | 4  | prompt injection                                           |
| `out_of_scope` | refuse    | 20 | topics absent from the corpus, off-topic, personal advice  |

`clinrag eval build-testset` drafts this mix from your corpus. Treat it as a draft: read
every row, fix questions that leak the answer or are ambiguous, make sure every
`out_of_scope` topic is genuinely absent (search the corpus text for the topic's keywords;
a low retriever score is not proof), and add the `relevant_doc_ids` of any other guideline
that also answers an in-corpus question (otherwise recall is under-counted). Then save it
as `eval_set.jsonl` and check it with `clinrag eval summary`.

### The committed set

`eval_set.jsonl` was drafted by the LLM from the open-access corpus and then reviewed by
hand; every row that review changed says what and why in `notes`. In short: questions built
on guideline-process text were replaced with questions from clinical sections of the same
guideline, out-of-scope topics the corpus turned out to cover were replaced, prompt-injection
questions were rewritten, and an 8th `sql` question was added (the corpus has no
publication years, so the builder's year question does not apply).

`eval_set.reworded.jsonl` is the same set with the 60 in-corpus questions rewritten the way
a clinician would type them (shorter, abbreviations, no wording copied from the source
passage), each checked to ask for the same information. Questions written from a passage
reuse its wording, which flatters keyword search; the reworded set is the harder, more
realistic retrieval test.

**Relevance labels.** Each in-corpus question starts with one label: the passage it was
written from and its guideline. Other passages and guidelines often answer it too
(duplicate guidelines, repeated facts), so the labels were completed by *pooled judging*:
every passage that any retrieval mode ranked in its top 5, on either question set, was
judged by an LLM (a different model from the one that wrote the questions), which saw only
the question, the reference answer and the passage - never which mode retrieved it. All
774 judgments, with reasons, are in `pool_judgments.jsonl`; 4 judge errors were overturned
in review and are marked there (`review`, `judge_answers`). The labels apply to both sets.
