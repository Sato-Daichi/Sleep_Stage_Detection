# Nishika 睡眠段階の判定 〜”睡眠の深さを判別しよう”〜

## コンペ概要

- リンク：https://www.nishika.com/competitions/sleep/summary

- タスク：睡眠ポリグラフや被験者のメタデータから睡眠段階を予測する多クラス分類

## 結果

- __ソロ参加 → チーム参加 → 最終順位：8位（金メダル🥇）__

- Public：4位（0.844） → Private：8位（0.849650）

- __このリポジトリはチームマージ前にソロで参加していた時のコードであり、8位を取ったコードではありません__

- 当リポジトリのコードは銅メダル相当です

- [チームの解法](https://www.nishika.com/competitions/sleep/topics/445)

- LightGBMベースの解法とEfficientNetベースの解法に二分された

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
