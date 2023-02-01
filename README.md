# Nishika 睡眠段階の判定 〜”睡眠の深さを判別しよう”〜

## コンペ概要

- リンク：https://www.nishika.com/competitions/sleep/summary

- タスク：睡眠ポリグラフや被験者のメタデータから睡眠の深さ（睡眠段階）を予測する（多クラス分類）

- 最終順位：8位（金メダル🥇）

    - __このリポジトリはチームマージ前にソロで参加していた時のコードであり、8位を取ったコードではありません__

    - ソロで作成したコードは銅メダル相当です

    - Public：4位（0.844） → Private：8位（0.849650）

    - [チームの解法](https://www.nishika.com/competitions/sleep/topics/445)

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
