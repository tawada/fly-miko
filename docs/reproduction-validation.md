# 再現性の検証記録

検証日: 2026-09-16。

## 実施したこと

- 既存の`data/`・`runs/`をコピーせず、一時的な別のソース配置を作成。
- `prepare_reproduction.py --download`により、注釈TSV・結合Parquet・一覧CSV・上流LICENSEを公開URLから実際に取得。
- 取得元の全SHA-256が固定値と一致。
- 入力・出力登録表と結合を新規にコンパイル。`config/reproduction.json`の全compiledChecksumsが一致。
- ニューロン138,639、結合15,091,983、入力1,736、出力1,280を確認。
- `config/reproduction-initial.json`から初期チェックポイントを復元し、既存実行`random-head-bias-02`の`state.initial`の594値と完全一致を確認。
- 新規のデータだけで実際のLIF・MuJoCoを使った第1世代を実行し、チェックポイントから第2世代を再開。
- 第1世代は20試行、第2世代は新規4試行のみ計算されたことを確認。両世代の全10候補×2条件の動作ファイルが独立保存されることを確認。
- ダウンロード失敗・チェックサム不一致・既存ファイルの保護・初期パラメータ復元を自動テスト。

初期パラメータのハッシュ（little-endian float64の594要素）:

```text
17f052c199b1f9ec83a8e44fb517e0f4cd6bc7ac058568333248e5a01f238979
```

## 検証の範囲と未検証事項

- 上記の新規環境テストは、**データと実行フォルダが空のソース配置**を使用しました。Pythonは既存のPython 3.11環境と`requirements-sim.txt`の指定バージョンを使用しています。OS・Python依存の新規インストールを含む完全に独立した環境の検証ではありません。
- 学習と再開の動作確認は**1試行0.1秒**で実施しました。READMEの本学習は10秒で、初期姿勢・入出力・初期パラメータ・seedは同じです。新規環境で10秒×全世代を完走させたという意味ではありません。
- 開発環境にはDockerコマンドがないため、Dockerイメージの実ビルド・実行は未検証です。Dockerfileの必要ファイル、作業ディレクトリ、entrypointとマウント先を確認しています。
- OSやCPUが異なる場合の浮動小数点結果の完全一致は未検証です。環境差を超えて同じ最良方策・報酬が得られることは保証しません。
- 取得先ファイルの将来の可用性は保証できません。内容が変わった場合はチェックサム検査で停止します。
- 実ファイルの取得成功は、その用途のライセンス承認を意味しません。利用条件は`data-preparation.md`を参照してください。

## 手元で確認するコマンド

Python環境を用意した場合:

```bash
.venv/bin/python scripts/prepare_reproduction.py --verify-only
.venv/bin/python -m unittest discover -s tests -p test_reproduction.py -v
```

`--verify-only`は既存のコンパイル済みデータと初期設定を照合し、`data/reproduction/`の初期チェックポイントを用意します。学習済みの実行を更新しません。
