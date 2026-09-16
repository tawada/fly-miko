# データの出典・利用条件・固定版

必要データは公開元から取得できます。トークンや旧PCからのコピーは不要です。取得物をGitへ追加しないでください。技術的に取得できることと、再配布・商用利用の許諾は区別します。

## 取得先

| 対象 | 固定版・公開元 | 用途 |
| --- | --- | --- |
| 神経注釈TSV | [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations/tree/8587524c1748ce5ef2080822a2fc890fc03bf597)、commit `8587524c1748ce5ef2080822a2fc890fc03bf597` | 上行性・下行性ニューロンと左右の対応 |
| `Connectivity_783.parquet`・`Completeness_783.csv` | [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model/tree/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960)、commit `91bdd1e7dcf193f3e7ca5a8933497fcef63b7960` | 全脳結合とニューロン一覧 |
| 初期出力変換 | `config/reproduction-initial.json` | このプロジェクトの594個の初期パラメータ |

実際のダウンロードURL・全SHA-256・コンパイル後に期待するチェックサムは[`config/reproduction.json`](../config/reproduction.json)に保存しています。URLは最新版ではなくcommitに固定しています。

## 利用条件の確認結果（2026-09-16）

- [FlyWire公式のcitation guidelines](https://flywire.ai/guidelines)は、公開データを**CC BY-NC 4.0**で提供すると記載しています。[利用規約](https://flywire.ai/tos)と引用方法も確認してください。商用利用を許可する案内ではありません。
- 一方、関連する[接続データのZenodoレコード](https://zenodo.org/records/10676866)にはCC-BY-4.0が表示されています。今回の入力ファイルはそのZenodoファイルではなく、上表のGitHub固定版です。他のレコードの条件を無条件に適用しません。
- `flywire_annotations`のREADMEは公開v783データの注釈ダンプとしてTSVを配布していますが、固定版にはLICENSEがなく、2023年の公開後の注釈更新も含まれます。**このTSV単体の再配布許諾は未確認**です。公式の一般条件が、独立した更新分すべての許諾を保証するとは扱いません。
- `Drosophila_brain_model`のリポジトリにはMIT通知があります。その全文を保持していますが、MITコードの通知をFlyWire原データ全般の条件として扱いません。
- 初期出力変換はこのプロジェクトの数値計算結果で、MMDモデルや外部の神経注釈・接続表そのものではありません。プロジェクトの独自コード・数値設定の再利用ライセンスは未選択です。

取得コマンドは読者が公開元から直接取得する操作です。このGitリポジトリでの外部データの再配布や、利用条件の解決済みを意味しません。対象用途の許諾が必要な場合は公開元へ確認してください。

## 準備コマンド

Docker版の完全なコマンドはREADMEの手順3を参照してください。Python環境を用意した場合は次です。

```bash
.venv/bin/python scripts/prepare_reproduction.py --download
```

`--download`を明示した場合だけ不足する元ファイルを取得します。既存ファイルも毎回チェックサムを検査します。ダウンロード途中・不一致のファイルは正式な保存先へ公開しません。コンパイル済みmanifestがある場合は既存データを検証し、作り直しません。

手元のコンパイル済みデータだけを検証し、初期値ファイルを復元する場合:

```bash
.venv/bin/python scripts/prepare_reproduction.py --verify-only
```

`brain:prepare`単独は引き続きローカルTSVの処理専用です。必要なデータを揃える入口は`data:prepare`です。

## 保存先

- `data/brain/flywire-neuron-annotations.tsv`：元の注釈
- `data/brain/upstream/`：結合Parquet、一覧CSV、上流ライセンス
- `data/brain/compiled/`：`indptr.npy`、`posts.npy`、`weights.npy`、`neuron-ids.json`、`input-binding.json`、`output-binding.json`、`manifest.json`
- `data/reproduction/random-head-bias-02/initial.npz`と`config.json`：初期出力変換。これは実行再開用の`latest.npz`ではありません

データ準備は新規環境で行ってください。稼働中の実行に対してコンパイルやデータの入れ替えを行わないでください。
