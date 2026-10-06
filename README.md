# Text Summarization on CNN/DailyMail: TF-IDF vs Seq2Seq vs Bahdanau Attention

**Authors:** Skander Adam Afi, Mohamed Harzallah. NLP project, M2 IASD

This project compares three approaches to automatic news summarization on the
CNN/DailyMail dataset. All three are scored with ROUGE-1/2/L on the test split:

1. **TF-IDF extractive**: ranks the article's sentences and keeps the top 3.
2. **RNN Encoder-Decoder (LSTM seq2seq)**: abstractive, trained with teacher forcing.
3. **RNN Encoder-Decoder + Bahdanau attention**: the same model with additive attention over the encoder states.

## Project structure

```
.
├── SkanderAdamAfi_MohamedHarzallah.ipynb   # main notebook (all code + results)
├── requirements.txt
├── data/                                    # CNN/DailyMail splits (not versioned, ~1.3 GB)
│   ├── train.csv        (~287k articles)
│   ├── validation.csv   (~13k)
│   └── test.csv         (~11k)
├── report/
│   └── NLP Project.docx                     # written report
└── archive/                                 # older backups (not needed to run)
```

Each CSV has the columns `id`, `article` and `highlights`. The `highlights` column is the reference summary.

## How to run

```bash
pip install -r requirements.txt
jupyter notebook SkanderAdamAfi_MohamedHarzallah.ipynb
```

Run the cells from top to bottom. All settings are in the **Configuration** cell
(`TRAIN_SIZE`, `EPOCHS`, `VOCAB_SIZE`, …).

To run on Google Colab, upload the three CSVs and set `DATA_DIR = "/content/"`.
A GPU is strongly recommended. With the default settings (20k training articles,
15 epochs), training takes hours on a CPU.

## Notes

- Padding is excluded from the training loss (`sample_weight`).
- The models train with Adam (lr = 1e-3).
- Greedy decoding never emits `<unk>` or padding, and never repeats the previous word.
- Expected outcome: TF-IDF is a strong baseline on CNN/DailyMail because the
  highlights are largely extractive. The from-scratch RNNs need many epochs on a GPU to close the gap.
