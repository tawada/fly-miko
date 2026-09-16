# 学習データの準備と配布範囲

このリポジトリは学習データを配布しません。動作中の`random-head-bias-02`は既存のローカルデータを利用します。

## 必要なローカルデータ

既存環境では`data/brain/compiled/`に以下が必要です。

- `manifest.json`（取得版・チェックサム）
- `indptr.npy`、`posts.npy`、`weights.npy`（疎結合）
- `neuron-ids.json`
- `input-binding.json`、`output-binding.json`（感覚・出力ニューロン登録）

`Connectome`はファイルのチェックサムを起動時に照合します。これらのファイルはGit対象外であり、別PCへ移す場合もデータの利用・共有条件を確認してください。

## 登録表と取得制約

既存の登録表は、固定版の`flyconnectome/flywire_annotations`に含まれる注釈TSVから作成したものです。当該固定版の再配布ライセンスを確認できていないため、TSV・派生登録表を同梱せず、自動ダウンロードも無効にしました。

適切な利用権限のもとで既に保持している同じ版のTSVを`data/brain/flywire-neuron-annotations.tsv`に配置した場合に限り、次のコマンドで登録表を作れます。スクリプトは固定チェックサムを検査します。

```bash
npm run brain:prepare
```

注釈データを所有していない新規利用者向けの、許諾確認済みの代替取得経路は本リポジトリには含めていません。

## 結合のコンパイル

登録表を準備した後、次で結合をコンパイルできます。

```bash
.venv/bin/python -m simulation.connectome
```

結合の取得元は`philshiu/Drosophila_brain_model`の固定版で、同リポジトリのMIT通知を保持しています。取得ファイルのURL・SHA-256は`simulation/connectome.py`に記載しています。この通知を無関係なFlyWire注釈やモデルに適用していません。取得データそのものをこのGitリポジトリに再配布することもありません。

コンパイルは既存ファイルを更新するため、学習稼働中には実行しないでください。学習結果の公開・データの再配布は、今回のソース公開とは別に扱います。
