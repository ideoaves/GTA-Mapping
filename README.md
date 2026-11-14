# GTAMapping

GTA の配信動画の左下のミニマップから、配信者の位置を読み取って地図に描くツールです。
番地入りの地図とSIFTで照合して位置を算出します。

## Python 環境
opencv-python, numpy, torch, easyocr, yt-dlp, ffmpeg/ffprobe

## 使い方
自環境以外でテストしていないので、まず1本の動画を処理できるか試してください。1本あたり1時間以内で終わります。

### 1. 動画をダウンロードする

```
python -m yt_dlp --write-info-json -o "video/メンバーのid_day日.mp4" "https://www.youtube.com/watch?v=動画ID"
```

`video/miko_day3.mp4` のように、`{メンバーのid}_day{日}.mp4` の名前で保存します。同日に2つある場合は `miko2_day3.mp4` とします。

`--write-info-json` は動画IDの取得と時間調整に利用されるので必須ファイルです。

### 2. gtamapping.py を動かす

```
python gtamapping.py
```

- まだ解析していない動画を全部解析します。`gtamapping.py` のダブルクリックでも同じです。
- `python gtamapping.py 2` で2日目だけ、`python gtamapping.py miko_day3` で該当の動画だけを解析します。
- 起動すると進み具合のページ（`editor/progress.html`）がブラウザで開きます。解析が終わっても「Enter で閉じます」と出るまでページは使えます。

うまくいっていない経路ができた場合は、動画名を指定して動かすと解析し直します。ミニマップの形状が違う場合は以下のオプションを付けてください。
```
python gtamapping.py miko_day3 --rect
```
- `--rect`: ミニマップを四角として測り直す
- `--round`: ミニマップを丸として測り直す

アバターなどでスピードメーターが隠れていて、運転中も直線ばかりになる場合は、`probes/{名前}.json` の `map` に `"blind": true` を足してから動画名を指定して動かします。車か徒歩かを判定せず毎回両方で照合します（解析時間は数倍になります）。

開始時刻がずれている場合があります。自動ではゲーム内時計の分が切り替わる瞬間を探して合わせていて、誤差は最大3秒です。時計が読めない配信では、ずれを0秒として扱います。
気になったら、ゲーム内時計の分が切り替わる瞬間で動画を止め、そのとき地図の再生バーに出ている時刻との差（ゲーム内時計 − 再生バー）を、`site/tracks/{名前}.json` の `offset` に足して直します。

### 3. 表示用の情報を書く
`site/event.json` に、タイトル、日付、メンバーの表示名と色と職業などのグループを書きます。地図に表示するときに使います。
解析が終わると下書きが自動で作られるので書き換えます。（idは入力済みなので変更しないように！）

`group` が空のメンバーは市民にまとまります。

```json
{
 "title": "[イベント名]マップ",
 "days": ["YYYY-MM-DD", "YYYY-MM-DD", "YYYY-MM-DD"],
 "groupColors": {"役職名": "#表示するカラーコード"},
 "members": [
  {"id": "id", "name": "表示名", "color": "#表示するカラーコード", "group": "役職名"}
 ],
 "places": []
}
```

### 4. edit.py で微調整する
```
python edit.py
```
`edit.py`（ダブルクリックでも可）を動かすと、ブラウザで経路編集のページ（`editor/edit.html`）が開きます。実際の配信を見ながら、間違っている経路を手で直します。施設モードでは、地図に出す施設を置けます。（event.jsonのplacesに入ります）
最初の保存のときに保存先を聞かれるので、`site` フォルダを選んでください（Chrome / Edge）。

ミニマップが読めずに直線になっている所は、直線上に点を置いてダブルクリックで赤くし、「微調整」ボタンを押すと、その両側を動画から照合し直します（`editor/fine-tune.py`）。照合には `video/` の動画が必要です。1区間あたり10〜50秒かかります。
直前の経路は `probes/{名前}_hand.json` に残ります。

### ページの開き方
`gtamapping.py` や `edit.py` を動かしている間は http://localhost:8000/ がローカルサーバーになり、`site/index.html`（地図）もここから開けます。
なおただMAPのみを公開する場合は `site` フォルダのみを公開すればよいです。

## gtamapping.py が作るファイル
| ファイル | 中身 |
| --- | --- |
| `data/targets.json` | 配信の一覧（メンバー、動画ID、開始時刻） |
| `probes/{名前}.json` | 動画ごとの HUD・ミニマップ・時刻のずれの測定値 |
| `probes/map_features.npz` | GTAの地図の特徴点（初回の解析時に `data/map.jpg` から作られます。違う地図を使う場合削除してください） |
| `probes/progress.json` | progress.html用の進捗データ |
| `probes/{名前}_hand.json` | 微調整する直前の経路（fine-tune.py が作る） |
| `site/tracks/{名前}.json` | 経路 |
| `site/tracks.json` | メンバー・日ごとの経路の目次 |
| `site/event.json` | サイト表示用の情報 |

## クレジット

- 地図（`data/map.jpg`、`site/tiles/`）: [Postal Code Map & Minimap v1.3](https://forum.cfx.re/t/release-postal-code-map-minimap-new-improved-v1-3/147458) by Virus_City
- `data/hud.png`、`data/speedo.png`: 音乃瀬奏「[【 #holoGTA 】権力。【音乃瀬奏】#hololiveDEV_IS #ReGLOSS](https://www.youtube.com/watch?v=KkdbflZg_Yo)」の配信画面
- 着想: [にじGTAライバーMAP](https://ide-an.github.io/nijigta-livers-map/)（[技術的な話](https://github.com/ide-an/nijigta-livers-map/blob/main/doc/technical_detail.md)）
- 地図の表示: [Leaflet](https://leafletjs.com/) 1.9.4（BSD-2-Clause）
- 文字認識: [EasyOCR](https://github.com/JaidedAI/EasyOCR)（Apache-2.0）
