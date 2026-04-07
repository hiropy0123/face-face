# face-face

顔写真から **年齢 / 性別 / 人種カテゴリ** を統計的に推定する Streamlit アプリケーションです。
事前学習済みモデル([DeepFace](https://github.com/serengil/deepface))を使用し、Streamlit Community Cloud へのデプロイを前提としています。

> ⚠️ 本アプリの推定結果は機械学習による統計的予測であり、特定個人の属性を断定するものではありません。差別的な目的での利用を固く禁じます。「人種カテゴリ」は DeepFace が出力する分類ラベル(asian / white / black / middle eastern / indian / latino hispanic)に基づきます。

---

## 主な機能

- 画像アップロード(JPEG / PNG)
- **複数人物の同時検出**
- 各顔に bounding box(四角マーカー)を描画し、`age` / `gender` / `race` を併記
- 各属性の確率(信頼度)を可視化
- 完全日本語 UI

## 技術スタック

| 項目 | 採用技術 |
|---|---|
| フロントエンド | Streamlit |
| 推論モデル | DeepFace(age / gender / race の事前学習モデル) |
| 顔検出器 | OpenCV(`opencv-python-headless`)+ DeepFace 内蔵 detector |
| 画像処理 | Pillow / NumPy |
| デプロイ | Streamlit Community Cloud(Python 3.11) |

> Streamlit Cloud のメモリ制約(約 1GB)に収まるよう、TensorFlow CPU 版を使用し、不要な GPU 依存は持ち込みません。

## ディレクトリ構成(予定)

```
face-face/
├── app.py                  # Streamlit エントリーポイント
├── src/
│   ├── __init__.py
│   ├── analyzer.py         # DeepFace 推論ラッパ
│   ├── drawing.py          # bounding box / ラベル描画
│   └── i18n.py             # 日本語ラベル変換
├── assets/
│   └── sample/             # サンプル画像
├── docs/
│   └── spec.md             # 仕様書
├── requirements.txt        # Streamlit Cloud 用依存(バージョン固定)
├── runtime.txt             # python-3.11
├── .streamlit/
│   └── config.toml
├── .gitignore
└── README.md
```

## ローカル実行

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## デプロイ手順(Streamlit Community Cloud)

1. GitHub リポジトリを作成し本リポジトリを push
2. https://share.streamlit.io/ から New app
3. リポジトリ / ブランチ / `app.py` を指定
4. Python version は `runtime.txt` の `python-3.11` が自動採用される

## 仕様書

詳細は [docs/spec.md](docs/spec.md) を参照してください。

## ライセンス / 免責

- DeepFace のモデルライセンスに準じます。
- 本アプリの出力は娯楽・教育目的の参考値であり、いかなる意思決定にも利用しないでください。
