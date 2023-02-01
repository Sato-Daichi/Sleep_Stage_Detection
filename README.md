# Nishika 睡眠段階の判定 〜”睡眠の深さを判別しよう”〜

## コンペ概要

- [コンペリンク](https://www.nishika.com/competitions/sleep/summary)

- 最終順位：8位（金メダル🥇）

    - このリポジトリはチームマージ前にソロで参加していた時のコードであり、8位を取ったコードではありません

    - ソロで作成したコードは銅メダル相当です

- タスク：睡眠ポリグラフ（polysomnography: PSG）から睡眠の深さ（睡眠段階）を予測する（多クラス分類）

- LightGBMベースの解法とEfficientNetベースの解法に分かれた

## ディレクトリ構成

```
├ data
│   ├ edf_data
│   │   ├ *-PSG.edf
│   │   └ *-Hypnogram.edf
│   ├ importance_plot
│   │   ├ ...
│   │   └ exp〇〇.jpg
│   ├ data_explanation.xlsx
│   ├ sweetviz.html
│   ├ test_eeg_epochs.pickle
│   ├ test_epochs.pickle
│   ├ test_records.csv
│   ├ train_eeg_epochs.pickle
│   ├ train_epochs.pickle
│   └ train_records.csv
├ notebooks
│   ├ ...
│   └ tutorial.ipynb
├ submission
│   ├ sample_submission.csv
│   ├ ...
│   └ exp〇〇.csv
├ .gitignore
└ README.md
```
