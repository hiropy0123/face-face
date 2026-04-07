# face-face 仕様書

## 1. 目的
アップロードされた画像に写る人物の **年齢・性別・人種カテゴリ** を事前学習済みモデルで推定し、可視化する Web アプリケーションを Streamlit で提供する。

## 2. 用語
| 用語 | 定義 |
|---|---|
| age | 推定年齢(整数、DeepFace の回帰出力) |
| gender | `Man` / `Woman` の二値分類 |
| race | DeepFace の 6 クラス分類:`asian` / `white` / `black` / `middle eastern` / `indian` / `latino hispanic` |
| 顔 | OpenCV / DeepFace の検出器が顔として検出した bounding box 領域 |

## 3. ユースケース
1. ユーザーが画像をアップロードする
2. システムが画像内の **すべての顔** を検出する
3. 各顔について age / gender / race を推定する
4. 元画像上に bounding box とラベルを重ねて表示する
5. 各顔の詳細(確率付き)を一覧表示する

## 4. 機能要件

### 4.1 入力
- ファイル形式:JPEG / PNG
- 最大サイズ:10 MB(Streamlit デフォルト 200MB を `config.toml` で 10MB に縮小)
- 最大解像度:長辺 2000px を超える場合は自動リサイズ

### 4.2 推論
- DeepFace `analyze()` を使用
- `actions=['age', 'gender', 'race']`
- `enforce_face=False`(顔が無い画像でもクラッシュさせない)
- `detector_backend='opencv'`(Cloud で最も軽量・追加依存なし)
- 複数顔対応:`analyze()` は配列を返す前提で実装

### 4.3 出力
- **画像表示**:検出した各顔に緑色の矩形を描画。矩形上部に `Age: 28 / Man / asian` の形式でラベル表示。
- **詳細パネル**:検出された顔ごとに以下を表示
  - サムネイル(切り出し)
  - 推定年齢
  - 性別と確率
  - 人種カテゴリの確率(棒グラフ、`st.bar_chart` または `st.progress`)
- **エラー時**:顔が検出できなかった場合は警告メッセージ表示。

### 4.4 国際化(i18n)
- UI 文字列はすべて日本語
- DeepFace 出力ラベル → 日本語マッピング(`src/i18n.py`)
  - `Man` → 男性 / `Woman` → 女性
  - `asian` → アジア系、`white` → 白人系、`black` → 黒人系、`middle eastern` → 中東系、`indian` → インド系、`latino hispanic` → ラテン系
- 元の英語ラベルも併記(誤訳防止)

### 4.5 注意書き表示
- 画面上部に常時、推定の限界・差別禁止に関する注意書きを `st.info` で表示。

## 5. 非機能要件
| 項目 | 要件 |
|---|---|
| 起動時間 | Streamlit Cloud 上で初回 60 秒以内(モデル初回ダウンロード含む) |
| 推論時間 | 顔 1 つあたり CPU で 5 秒以内 |
| メモリ | ピーク 900MB 以下(Streamlit Cloud 1GB 制限内) |
| 可用性 | Streamlit Community Cloud の SLA に準ずる |
| セキュリティ | アップロード画像はサーバ側に永続化しない(メモリ上のみ) |

## 6. アーキテクチャ

```
[ User ]
   │  upload
   ▼
[ Streamlit (app.py) ]
   │
   ├─► src/analyzer.py ──► DeepFace.analyze()
   │                          │
   │                          ├─ 顔検出 (OpenCV)
   │                          ├─ Age model
   │                          ├─ Gender model
   │                          └─ Race model
   │
   ├─► src/drawing.py ──► OpenCV / Pillow で描画
   └─► src/i18n.py    ──► ラベル日本語化
```

## 7. モジュール依存(`requirements.txt` 想定)

Streamlit Community Cloud(Python 3.11)で動作確認済みの組み合わせとしてバージョン固定する。

```
streamlit==1.39.0
deepface==0.0.93
tf-keras==2.17.0
tensorflow-cpu==2.17.0
opencv-python-headless==4.10.0.84
Pillow==10.4.0
numpy==1.26.4
pandas==2.2.3
```

> ポイント:
> - **`tensorflow-cpu`** を使用(GPU 版は Cloud で不要かつ重い)
> - **`opencv-python-headless`** を使用(GUI 依存を排除)
> - DeepFace 0.0.93 以降は `tf-keras` 明示インストールが必要
> - `runtime.txt` に `python-3.11` を記載

## 8. ファイル別仕様

### 8.1 `app.py`
- ページ設定(`st.set_page_config(page_title="顔分析アプリ", layout="wide")`)
- ヘッダー / 注意書き
- `st.file_uploader`(jpg, jpeg, png)
- アップロード後:`analyzer.analyze_image()` 呼び出し
- 結果画像 + 詳細パネル表示
- 例外ハンドリング(顔未検出 / モデル読み込み失敗)

### 8.2 `src/analyzer.py`
```python
def analyze_image(image: np.ndarray) -> list[FaceResult]:
    """DeepFace.analyze を呼び、顔ごとの結果リストを返す"""
```
- 結果は dataclass `FaceResult` に正規化
  - `region: dict[str, int]`(x, y, w, h)
  - `age: int`
  - `gender: str`, `gender_confidence: float`
  - `dominant_race: str`, `race: dict[str, float]`

### 8.3 `src/drawing.py`
```python
def draw_faces(image: np.ndarray, faces: list[FaceResult]) -> np.ndarray
```
- OpenCV `cv2.rectangle` + `cv2.putText`(日本語フォントは扱いが難しいため、画像内ラベルは英語、画面下の詳細を日本語にする方針)

### 8.4 `src/i18n.py`
- 性別 / 人種ラベルの日本語マッピング辞書を提供

### 8.5 `.streamlit/config.toml`
```toml
[server]
maxUploadSize = 10

[theme]
base = "light"
```

## 9. テスト計画
- **手動テスト**:
  - 1人の正面顔
  - 複数人(3〜5人)の集合写真
  - 顔が写っていない画像
  - 横向き / マスク着用 / 低解像度
- **回帰テスト**:`assets/sample/` 配下にサンプルを置き、起動時のスモークテスト用にする

## 10. リスクと対応
| リスク | 対応 |
|---|---|
| Cloud のメモリ超過 | `tensorflow-cpu` 採用、画像リサイズ、`@st.cache_resource` でモデル単一インスタンス化 |
| 初回モデルダウンロードのタイムアウト | `@st.cache_resource` で起動時に DeepFace を warm-up |
| 顔未検出 | エラーではなく警告で UX を維持 |
| 推定精度の誤解 | 注意書きを常時表示、確率を必ず併記 |
| 倫理的懸念(人種推定) | 用途を明示し、差別禁止の文言を表示 |

## 11. 実装ステップ
1. `requirements.txt` / `runtime.txt` / `.streamlit/config.toml` 整備
2. `src/analyzer.py`:DeepFace ラッパ + dataclass 定義
3. `src/drawing.py`:bounding box 描画
4. `src/i18n.py`:日本語ラベル
5. `app.py`:UI 組み立て
6. ローカル動作確認(`streamlit run app.py`)
7. GitHub リポジトリ作成 → push
8. Streamlit Community Cloud にデプロイ
9. サンプル画像で受け入れテスト

## 12. 未決事項(要相談)
- なし(現時点で要件は確定)
