# GTAMapping

GTA の配信動画の左下のミニマップから、配信者の位置を読み取って地図に描くツールです。
番地入りの地図とSIFTで照合して位置を算出します。

# Python 環境
opencv-python, numpy, torch, easyocr, yt-dlp, ffmpeg/ffprobe

# 使い方
自環境以外でテストしていないので、まず1本の動画を処理できるか試してください。1本あたり1時間以内で終わります。

## 1. 動画をダウンロードする
yt_dlpを使うのが良いです。`--write-info-json` は動画IDの取得と時間調整に利用されるので必須ファイルです。
```
python -m yt_dlp --write-info-json -o "video/メンバーのid_day日.mp4" "https://www.youtube.com/watch?v=動画ID"
```

`video/miko_day3.mp4` のように、`{メンバーのid}_day{日}.mp4` の名前で保存します。同日に2つある場合は `miko2_day3.mp4` とします。

Twitch のアーカイブも同じ方法で落とせば、地図からTwitchの動画として再生できます。開始時刻は画面上部の日時から逆算します。

## 2. gtamapping.py を動かす
`gtamapping.py` のダブルクリックで動画を全部解析します。

`python gtamapping.py 2` で2日目だけ、`python gtamapping.py miko_day3` で該当の動画だけを解析します。

うまくいっていない経路ができた場合は、動画名を指定して動かすと解析し直します。ミニマップの形状が違う場合は以下のオプションを付けてください。
- `--rect`: ミニマップを四角として測り直す
- `--round`: ミニマップを丸として測り直す

直線ばかりになる場合は、`probes/{名前}.json` の `map` に `"blind": true` を足してから動画名を指定して動かします（解析時間は数倍になります）。

開始時刻がずれている場合があります。自動ではゲーム内時計の分が切り替わる瞬間を探して合わせていて、誤差は最大3秒です。
さらに、サーバー再起動前のアナウンスでも合わせます。`gtamapping.py` 上部の `RESTART_AT` に再起動の時刻を書いておくとその40分前からの告知を探し、日ごとに時計と告知のうち多くの配信で読めた方を優先します。告知を使わないサーバーでは `RESTART_AT = ''` にしてください。

どちらも読めない配信は、ずれを0秒として扱います。手動で調整も可能です。ゲーム内時計の分が切り替わる瞬間で動画を止め、そのとき地図の再生バーに出ている時刻との差（ゲーム内時計 − 再生バー）を、`site/tracks/{名前}.json` の `offset` に足して直してください。

## 3. 表示用の情報を書く
解析が終わると`site/event.json` の下書きが自動で作られるので書き換えます。（idは入力済みなので変更しないように！）

タイトル、日付、メンバーの表示名と色と職業などのグループを書きます。地図に表示するときに使います。

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
## 4. 微調整

`edit.py`をダブルクリックします。

ブラウザで経路編集のページが開きます。実際の配信を見ながら、間違っている経路を手で直し保存します。保存先を聞かれるので、`site` フォルダを選んでください。

また補助的ではありますが、直線になっている箇所は直線をダブルクリックで赤くし「微調整」ボタンを押しても良いです。1区間あたり10〜50秒かかります。

施設モードでは、地図に出す施設を置けます。

サーバー独自のワープ先（スタック救済コマンドの移動先など）があれば、そこに `warp` という名前で施設を置いてください。遠くからそこへ飛んだときに経路の線を切ります(gtamapping.pyを再実行することで反映されます)。

## 5. 公開
site/の中身が実際のMAPサイトです。これを公開すれば完成です。

# その他
## gtamapping.py が作るファイル
| ファイル | 中身 |
| --- | --- |
| `data/targets.json` | 配信の一覧（メンバー、動画ID、開始時刻） |
| `probes/{名前}.json` | 動画ごとの HUD・ミニマップ・時刻のずれの測定値 |
| `probes/map_features.npz` | GTAの地図の特徴点（初回の解析時に `data/map.jpg` から作られます。違う地図を使う場合削除してください） |
| `probes/progress.json` | progress.html用の進捗データ |
| `site/tracks/{名前}.json` | 経路 |
| `site/tracks.json` | メンバー・日ごとの経路の目次 |
| `site/event.json` | サイト表示用の情報 |

## クレジット

- 地図（`data/map.jpg`、`site/tiles/`）: [Postal Code Map & Minimap v1.3](https://forum.cfx.re/t/release-postal-code-map-minimap-new-improved-v1-3/147458) by Virus_City
- `data/hud.png`、`data/speedo.png`: 音乃瀬奏「[【 #holoGTA 】権力。【音乃瀬奏】#hololiveDEV_IS #ReGLOSS](https://www.youtube.com/watch?v=KkdbflZg_Yo)」の配信画面
- 開発・テストに使った配信: [Video_credit.md](Video_credit.md)
- 着想: [にじGTAライバーMAP](https://ide-an.github.io/nijigta-livers-map/)（[技術的な話](https://github.com/ide-an/nijigta-livers-map/blob/main/doc/technical_detail.md)）
- 地図の表示: [Leaflet](https://leafletjs.com/) 1.9.4（BSD-2-Clause）
- 文字認識: [EasyOCR](https://github.com/JaidedAI/EasyOCR)（Apache-2.0）
