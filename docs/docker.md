# Dockerで学習する

DockerfileはCPU学習用です。Python 3.11公式イメージに科学計算ライブラリと
MuJoCoの実行ライブラリを導入します。ホストでPythonやvenvを用意する必要はありません。
デモのNode.jsサーバーや動画生成用ブラウザーはこのイメージに含みません。

## ビルド

コピー先のPCでDockerを用意し、プロジェクト内で実行します。

```bash
cd ~/docker_kekkingu/workspace/fly-miko
docker build -t fly-miko-train .
```

環境はコンテナの`/opt/fly-miko-venv`に作ります。ホストの`.venv`は使いません。
`ensurepip`を含むPython公式イメージを使うため、ホストの`python3.11-venv`不足に依存しません。
依存バージョンは`requirements-sim.txt`で固定し、ビルド時に依存検査と物理計算の起動検査を行います。

モデル・結合データ・学習結果は`.dockerignore`でビルド対象から外します。
実行時にプロジェクトをマウントし、元のデータとチェックポイントを読み書きします。

## 保存済みの学習を再開

Linux／WSLのシェルで、プロジェクトのディレクトリから実行します。

```bash
docker run --rm -it --init \
  --user "$(id -u):$(id -g)" \
  --mount type=bind,source="$PWD",target=/workspace/fly-miko \
  fly-miko-train \
  --name random-head-bias-02 \
  --task head-height \
  --resume --iterations 20 \
  --population 10 --workers 1 \
  --seconds 10
```

`--iterations`は追加数ではなく総世代数です。保存済み世代より大きく指定してください。
`--user`はコピー先PCのユーザーでファイルを作成するための指定です。
`NUMBA_CACHE_DIR`などは書き込み可能な`/tmp`に置きます。
学習結果はホストの`runs/training/`に残り、コンテナを削除しても消えません。
Ctrl+Cで中断できます。最後に完了して保存した世代から再開します。

`data/brain/compiled/`と`runs/training/random-head-bias-02/`が必要です。
ソース・設定・データの版とチェックポイントが一致することも再開時に検査します。
ソースや`requirements-sim.txt`を変更した場合はイメージをビルドし直してください。

新規学習では`--resume`を外し、未使用の`--name`を指定します。[現在の学習シナリオ](population-search.md)も参照してください。
CLIのヘルプだけを表示する場合は以下です。

```bash
docker run --rm fly-miko-train --help
```

## 別PCへコピーするとき

学習を停止してからコピーします。Git管理外の`data/`と`runs/`も含めてください。
標準の3D表示にキャラクターモデルは不要です。第三者への配布では各データの許諾を確認し、モデルを含むフォルダ全体を共有しないでください。
`.venv`や`node_modules`はコピー先の実行環境として再利用しません。

```bash
scp -r workspace/fly-miko gawa-mokoko:~/docker_kekkingu/workspace/
```

コピー先で`Permission denied`になる場合は、コピー先にログインして所有者と権限を確認します。

```bash
ssh gawa-mokoko
ls -ld ~/docker_kekkingu/workspace/fly-miko \
  ~/docker_kekkingu/workspace/fly-miko/{src,tests,third_party}
```

このプロジェクトを自分の作業フォルダーとして使う場合、所有者と自分の書き込み権限を修正します。

```bash
sudo chown -hR "$(id -u):$(id -g)" ~/docker_kekkingu/workspace/fly-miko
chmod -R u+rwX ~/docker_kekkingu/workspace/fly-miko
exit
```

その後、コピー元で`scp`を再実行します。Docker学習時も上記の`--user`指定を使います。

所有者と権限が正しくても共有フォルダー上で保存時の置換が失敗する場合があります。
保存処理は一時的なアクセス拒否を最大10回再試行し、失敗時は既存ファイルを保持します。
起動前の検査だけ行う場合は、上記の`docker run`に続く学習引数を
`--name random-head-bias-02 --check-storage`へ置き換えます。
検査が継続して失敗する場合は、共有フォルダーではなくPCのローカルディスクへコピーして実行してください。
