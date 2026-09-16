# fly-miko

固定LIF神経回路と18関節の物理身体を接続し、ランダム初期姿勢2通りから頭を高く保つ制御を学習します。ここでは **random-head-bias-02** のデータ取得・初期値の復元・学習・表示を説明します。MMDモデルは不要です。

学習するのは **32特徴量×18関節＝576個の重みと、18個のバイアス、合計594個**です。567個ではありません。式は `a = tanh(W f + b)`。固定された脳内の結合重みとは別の、脳出力から関節指令への変換です。

## 前提と再現範囲

- 以下のDockerコマンドはLinux／WSL2のシェル用です。Dockerが必要です。Dockerをsudoなしで使える環境では`sudo`を省略できます。
- データ準備・学習はDocker内で実行でき、ホストのPython・venv・npmは不要です。ブラウザ表示・動画撮影を行う場合だけ、後半のNode.js環境も用意します。
- データは公開元から取得します。APIトークンと手元の古い学習結果は不要です。元データ約150MBに加えて、コンパイル済みデータ、Dockerイメージ、学習キャッシュ・世代別動作を保存する空き容量が必要です。後者は学習量に応じて増加します。
- 本手順はデータの版・接続表・594個の初期値・seed・評価条件を揃えます。異なるCPUや数値ライブラリでの結果のビット単位一致、歩行や起立の獲得は保証しません。

**データの利用条件を先に確認してください。** FlyWireの公開データには非商用等の条件があります。また、今回使う固定版注釈TSVの個別の再配布許諾は未確認です。公開URLから取得できることと、自由に再配布・商用利用できることは別です。[出典・利用条件・SHA-256](docs/data-preparation.md)を参照してください。外部データ本体はこのGitリポジトリには同梱していません。

## 1. cloneして作業フォルダへ移動

```bash
git clone https://github.com/tawada/fly-miko.git
cd fly-miko
pwd
ls Dockerfile simulation/train.py scripts/prepare_reproduction.py
mkdir -p data runs renders
```

以降のコマンドはすべて、この`fly-miko`フォルダで実行してください。

## 2. Dockerイメージを作る

```bash
sudo docker build -t fly-miko-train .
sudo docker run --rm fly-miko-train --help
```

イメージ内にPython 3.11と依存パッケージを導入します。ホスト側の`ensurepip`や`.venv`は使用しません。

## 3. 公開元からデータを取得し、初期パラメータを復元

```bash
sudo docker run --rm -it --init \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source="$PWD/data",target=/workspace/fly-miko/data \
  --entrypoint python \
  fly-miko-train scripts/prepare_reproduction.py --download
```

このコマンドは次を順に行います。

1. 固定版の注釈TSV、結合Parquet、ニューロン一覧CSV、上流ライセンスを公開元からダウンロード。
2. 全ファイルのSHA-256を照合し、入力・出力登録表と疎結合を作成。
3. 出来上がった結合・登録表を、元の実行のチェックサムと照合。
4. 公開した初期値JSONから、594個の数値が同じ初期チェックポイントを復元。

完了すると、138,639ニューロン・15,091,983結合・入力1,736個・出力1,280個が確認され、以下ができます。

```text
data/brain/compiled/                         # 脳・登録表・manifest
data/reproduction/random-head-bias-02/
  initial.npz                               # 学習開始用の594個の初期値
  config.json
```

既に正しいデータがあれば再利用します。不一致ファイルを勝手に上書きせず停止します。途中で通信が失敗したら同じコマンドを再実行してください。この段階では`runs/training/random-head-bias-02/`は作りません。

## 4. 同じ初期値から新規学習

初回は **`--resume`を付けません**。`--init-from`で手順3の初期値を指定します。

```bash
sudo docker run --rm -it --init \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source="$PWD/data",target=/workspace/fly-miko/data,readonly \
  --mount type=bind,source="$PWD/runs",target=/workspace/fly-miko/runs \
  fly-miko-train \
  --name random-head-bias-02 \
  --init-from data/reproduction/random-head-bias-02/initial.npz \
  --iterations 10 --population 10 --workers 1 --seconds 10 --seed 35
```

初期値は`config/reproduction-initial.json`に保存した、このプロジェクトの学習で生成した出力変換の数値です。元の実行は旧チェックポイントからこれを引き継いでいましたが、この手順ではその旧チェックポイントは不要です。3Dモデルや脳内結合のデータではありません。

`--init-from`を省くと標準の乱数初期重みになるため、元の`random-head-bias-02`と同じ初期条件にはなりません。

1候補を2通りの初期姿勢で各10秒評価し、頭の中心の高さの積分（m・秒）の平均を目的関数にします。初期身体seedは3035・3036、脳の乱数seedは1035・1036で実行内固定です。首は胴体に対して中立角、肘の基準角は0radです。

1世代は最良1・小さい変異4・大きい変異3・新規ランダム2の計10候補で、重複を排除します。5世代停滞するごとに変異幅を拡大します（最大4倍）。通常は初世代20試行、その後は各世代18試行を新たに計算します。旧方式の実行は同じ`--resume`で引き継げます。評価条件を照合し、旧設定・チェックポイントを`before-mutation-v2/`に保存して、次世代から切り替えます。**シミュレーションの10秒は実時間の10秒ではありません。**

## 5. 中断後に再開

先にチェックポイントがあることを確認します。

```bash
ls runs/training/random-head-bias-02/latest.npz
```

再開では`--init-from`を外し、**`--resume`を付けます**。

```bash
sudo docker run --rm -it --init \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source="$PWD/data",target=/workspace/fly-miko/data,readonly \
  --mount type=bind,source="$PWD/runs",target=/workspace/fly-miko/runs \
  fly-miko-train \
  --name random-head-bias-02 --resume \
  --iterations 20 --population 10 --workers 1 --seconds 10 --seed 35
```

`--iterations 20`は追加20世代ではなく、合計20世代までという意味です。再開時は学習開始時と同じソース・依存・データ・設定を使ってください。ソースを変更したイメージでの再開は、互換性検査で拒否される場合があります。

## 6. 結果の保存先

`data/`と`runs/`を個別にマウントしているので、コンテナを削除してもホスト側に結果が残ります。

```text
runs/training/random-head-bias-02/
  latest.npz                 # 再開用。評価キャッシュ等も一緒に保持
  progress.json              # 現在の進捗
  history.json               # 各世代の全候補の評価
  best.json / best-second.json
  evaluation-cache/          # 全試行の評価値・動作
  generations/0001/          # 指定世代の全10候補×2初期状態の動作・パラメータ
```

第1〜5世代と第10・20・30…世代を、動画化用に独立保存します。詳細・JSON取り出しコマンドは[学習シナリオ](docs/population-search.md)にあります。別PCで再開する場合は元の学習を止めてから、実行フォルダ全体をコピーしてください。

## 7. デモ表示・動画撮影

現在のDockerfileは学習専用です。表示・撮影はホスト側にNode.js 20以上を用意し、同じ`fly-miko`フォルダで実行します。

```bash
npm ci
npm run demo:preview
```

`http://127.0.0.1:8000/demo/training.html`を開きます。最初の2試行の評価が終わると、物理身体の再生記録が表示されます。ポート8000はコンテナの学習処理には不要です。デモサーバーは`0.0.0.0:8000`で待機します。

MP4を生成する場合はブラウザもインストールします。Linuxのシステム依存導入には管理者権限が必要な場合があります。

```bash
npx playwright install chromium
npx playwright install-deps chromium
npm run training:render
```

動画は`renders/`に保存します。保存済み物理軌跡の再生で、撮影中に再学習しません。日本語フォントはOSのものを利用でき、キャラクターモデルは不要です。

全脳の20msごとの発火を表示したい場合は次を実行し、デモの「最新結果を読込」を押します。

```bash
sudo docker run --rm -it --init \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source="$PWD/data",target=/workspace/fly-miko/data,readonly \
  --mount type=bind,source="$PWD/runs",target=/workspace/fly-miko/runs \
  --entrypoint python \
  fly-miko-train -m simulation.activity --name random-head-bias-02 --scenario best
```

発火は独自の2D一覧・時系列で表示します。MMDと`fly-brain-vis`による解剖学的3D表示は公開版に含めていません。

## Dockerを使わない場合

Python 3.11、Node.js 20以上を用意し、同じデータ準備・初期値を使います。

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-sim.txt
npm ci
npm run data:prepare
npm run train -- --name random-head-bias-02 \
  --init-from data/reproduction/random-head-bias-02/initial.npz \
  --iterations 10 --population 10 --workers 1 --seconds 10 --seed 35
```

Debian/Ubuntuで`ensurepip`がない場合は`sudo apt install python3.11-venv`を実行してください。Pythonの版が異なるOSではDockerの手順を使ってください。

## 困ったとき・検証範囲

- `No module named simulation`：作業フォルダを確認。上記のように`data/`と`runs/`だけをマウントし、イメージ内のプログラムを隠さないでください。
- `No checkpoint to resume`：初回は手順4です。別PCから再開するなら実行フォルダ全体が必要です。
- `Run already exists`：既存結果を再開するか、保管してから新規実行してください。データ準備時に実行フォルダを先に作る必要はありません。
- `Checksum mismatch`：異なる版・破損したファイルを自動上書きしません。対象を退避して取得し直してください。
- 保存権限エラー：ホストの`data/`・`runs/`を自分が所有し、書き込めることを確認。起動時は上記の`--user`を指定してください。

空のデータディレクトリから公開元の実ファイルを取得・コンパイルし、元データの全チェックサム一致を確認する検証を用意しています。実行した範囲と限界は[再現性の検証記録](docs/reproduction-validation.md)を参照してください。Dockerそのもののビルド・実行は開発環境にDockerがないため未検証です。

独自コード・初期数値設定の再利用ライセンスは未選択です。[LICENSES.md](LICENSES.md)・[NOTICE.md](NOTICE.md)を参照してください。外部データ・モデル・依存本体・生成結果はGitに含めず、`npm run repo:check`で配布対象を検査します。
