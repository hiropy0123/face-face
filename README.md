---
title: Face Face
emoji: 🙂
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

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
| デプロイ | **Hugging Face Spaces**(Streamlit SDK / CPU Basic / 16GB RAM) |

> 当初 Streamlit Community Cloud を想定していましたが、DeepFace の age/gender/race モデル(各 ~537MB、合計 ~1.6GB)が Cloud の 1GB メモリ制限に収まらないため、Hugging Face Spaces(CPU Basic 無料枠 16GB RAM)に変更しました。

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

## デプロイ手順(Hugging Face Spaces)

1. https://huggingface.co/new-space で新しい Space を作成
   - **SDK**: Streamlit
   - **Hardware**: CPU basic (free)
   - 公開/非公開を選択
2. 作成された Space の git リポジトリに本リポジトリの内容を push
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/face-face
   git push space main
   ```
3. Space が自動でビルド・起動します(初回は DeepFace モデルのダウンロードに数分)
4. 設定は本 README 冒頭の YAML frontmatter で管理(`sdk`, `sdk_version`, `app_file` 等)

> 📝 `packages.txt` の apt 依存(`libgl1`, `libglib2.0-0t64`)は HF Spaces でもそのまま有効です。

## 仕様書

詳細は [docs/spec.md](docs/spec.md) を参照してください。

## ライセンス / 免責

- DeepFace のモデルライセンスに準じます。
- 本アプリの出力は娯楽・教育目的の参考値であり、いかなる意思決定にも利用しないでください。
