# fly-miko

固定のLIF神経回路と18関節の物理身体を接続し、出力変換の594パラメータを探索します。公開版の対象は **random-head-bias-02**（ランダム初期姿勢、首・関節の中立基準）です。3D表示は自作の球・カプセル・箱による物理形状で、MMDは使用しません。

## 配布内容

[LICENSES.md](LICENSES.md)と[NOTICE.md](NOTICE.md)に権利表示・外部依存の扱いを記載しています。独自コードへの再利用ライセンスは未選択です。

Gitにはソース・設定・テスト・文書・ライセンス通知だけを含めます。モデル、外部神経データ、派生登録表、外部可視化実装、学習結果、画像・動画、依存パッケージ本体は含めません。

## 準備

Python 3.11、Node.js 20以上を使用します。

```bash
npm ci
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-sim.txt
```

Debian/Ubuntuで`ensurepip`がない場合は`python3.11-venv`をインストールするか、[Docker手順](docs/docker.md)を利用してください。

**ソースだけでは学習用データは揃いません。** 注釈データと登録表は配布許諾未確認のため同梱・自動取得を行いません。既存のローカルデータを使う手順と、必要ファイルは[データ準備](docs/data-preparation.md)を参照してください。データ準備コマンドは稼働中の学習と同時に実行しないでください。

## 学習

新規実行（保存先がまだ存在しない場合）:

```bash
npm run train -- --name random-head-bias-02 \
  --iterations 10 --population 10 --workers 1 --seconds 10
```

既存の実行を再開:

```bash
npm run train -- --name random-head-bias-02 --resume \
  --iterations 20 --population 10 --workers 1 --seconds 10
```

`--iterations`は追加世代数ではなく到達する総世代数です。各候補を固定したランダム初期姿勢2通りで10秒ずつ評価し、頭の高さの積分の平均を目的関数にします。首は胴体に対して中立角、肘の基準角は0radです。首は独立した学習関節ではありません。

1世代は最良1・新規ランダム2・前世代上位5・残りからランダム2の計10候補です。初世代は初期候補1＋ランダム9。計算済み評価は再利用します。脳出力32特徴量から18関節への576重み＋18バイアスを学習します。

第1〜5世代と第10・20・30…世代は全10候補×2初期状態の動作・パラメータを世代別に独立保存します。学習・キャッシュ・動画用データの詳細は[学習シナリオ](docs/population-search.md)を参照してください。

## デモと動画

```bash
npx playwright install chromium
# Linuxで必要な場合
npx playwright install-deps chromium
npm run demo:preview
```

`http://127.0.0.1:8000/demo/training.html`で`random-head-bias-02`の身体・評価推移・脳出力を確認できます。サーバーは`0.0.0.0:8000`で待機し、Dockerでは`-p 8000:8000`を指定します。初期評価が完了すると最初の再生記録が表示されます。

```bash
npm run training:render
npm run brain:record-activity -- --name random-head-bias-02 --scenario best
```

動画は`renders/`に保存します。全脳発火記録は20msごとに元軌跡と照合し、身体の再生時刻と同期した2D一覧・時系列で表示します。解剖学的位置を示す表示ではありません。最良記録を更新したら発火記録も再作成してください。

## 検査と共有

```bash
npm run repo:check
npm run bridge:test
.venv/bin/python -m unittest discover -s tests -v
# 実データと最初の評価記録がある場合
npm run training:check
npm pack --dry-run
```

`assets/`・`downloads/`・`data/`・`runs/`・`renders/`・仮想環境等はGit、npmパッケージ、Dockerビルドから除外します。`.gitignore`は`scp -r`に適用されないため、プロジェクト全体を第三者へそのまま配布しないでください。
