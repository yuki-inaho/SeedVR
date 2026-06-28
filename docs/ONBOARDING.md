# SeedVR2-3B — cu128 / Blackwell オンボーディング

このブランチ(`cu128`)は、**NVIDIA Blackwell (sm_120)** GPU + 比較的古い OS（glibc 2.31, nvcc 無し）でも
SeedVR2-3B 推論を動かすための **uv ベースのモダン cu128 スタック**です。
upstream README の忠実 pin（torch2.4+cu121 / flash-attn / apex）はこの環境では動かないため、
attention=torch SDPA、normalization=純torch、DiT=bf16 に置き換えています。

## 0. 対象環境 / 前提

- GPU: Blackwell (sm_120) など。検証実機: RTX PRO 4500 Blackwell (32GB)
- driver: CUDA 12.8 対応（例: 580.x）
- OS: Ubuntu 20.04 (glibc 2.31, libstdc++ max GLIBCXX_3.4.28), **nvcc 不要**
- [uv](https://docs.astral.sh/uv/) が導入済みであること

> なぜ cu128 か: Blackwell sm_120 は torch cu121/cu124 には無く、**torch ≥2.7 の cu128** が必要。
> また prebuilt の flash-attn は glibc≥2.32、apex は GLIBCXX_3.4.29 を要求しこの OS で import 不可、
> nvcc 無しでソースビルドも不可。→ flash-attn を **torch SDPA**、apex を **純torch RMSNorm/LayerNorm** で代替。

## 1. セットアップ（uv）

```bash
cd SeedVR
uv sync          # pyproject.toml + uv.lock から .venv を再現（torch 2.8.0+cu128 等）
```

`uv sync` は torch/torchvision を `pytorch-cu128` index（`https://download.pytorch.org/whl/cu128`）から解決します。
flash-attn / apex は依存に含まれません（上記理由）。

## 2. color_fix.py の取得（リポジトリには含めない）

color fix を使う場合のみ、upstream の指示どおり手動で配置します（ライセンス上リポジトリには同梱しません）。

```bash
curl -fsSL https://raw.githubusercontent.com/pkuliyi2015/sd-webui-stablesr/master/srmodule/colorfix.py \
  -o projects/video_diffusion_sr/color_fix.py
```

未配置でも動作します（その場合は `use_colorfix=False` で色補正なし）。

## 3. 重みの取得

```bash
uv run python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="ByteDance-Seed/SeedVR2-3B",
    local_dir="ckpts",
    allow_patterns=["ema_vae.pth", "seedvr2_ema_3b.pth"],
)
PY
```

- `ckpts/ema_vae.pth` (≈957MB) / `ckpts/seedvr2_ema_3b.pth` (≈13GB, fp32)
- `pos_emb.pt` / `neg_emb.pt` はリポジトリ同梱（CWD から読まれる）
- `ckpts/` は `.gitignore` 済み

## 4. 推論の実行

```bash
# 入力フォルダ（画像 or 動画）を用意。例: 低解像度画像 1 枚を smoke_in/ に置く
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=$PWD uv run torchrun --nproc-per-node=1 --master_port=29580 \
  projects/inference_seedvr2_3b.py \
  --video_path smoke_in --output_dir smoke_out \
  --res_h 128 --res_w 128 --sp_size 1
```

- **`PYTHONPATH=$PWD` 必須**（torchrun 直起動だと repo root が sys.path に入らず `data` 等が未解決）
- 画像入力時は **`--sp_size 1`** 必須。入力フォルダ内の各ファイルを処理し同名で `output_dir` に出力。
- `--res_h/--res_w` で出力解像度。VRAM に応じて調整。

## 5. cu128 スタックで upstream から変更した点

| ファイル | 変更 | 理由 |
|---|---|---|
| `configs_3b/main.yaml` | `vid_out_norm/qk_norm/norm: fusedrms→rms`, `txt_in_norm: fusedln→layer` | apex(FusedRMSNorm/FusedLayerNorm) を回避し純torch normへ |
| `models/dit_v2/attention.py`, `models/dit/attention.py` | `flash_attn_varlen_func` を **SDPA版 drop-in 関数**に置換 | flash-attn 排除（glibc 非互換 / ビルド不可） |
| `projects/video_diffusion_sr/infer.py` | DiT を `dtype=torch.bfloat16` で GPU 配置 | 3B を 32GB に収める（fp32 ~12GB→~6GB） |
| `projects/inference_seedvr2_3b.py` | 画像入力分岐に `fps_lists.append(out_fps)` | upstream バグ: 画像入力で出力が保存されない問題の修正 |
| `pyproject.toml` / `uv.lock` | uv プロジェクト化（cu128 index 明記） | `uv sync` で再現可能に |

## 6. 既知の制約

- **VRAM 32GB**: upstream 想定は H100 80GB。本スタックでは小解像度・少フレーム向き。大入力は OOM の可能性
  → `--res_*` を下げる / フレーム数を絞る / `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
- **数値差**: apex融合norm→純torch、flash-attn→SDPA、DiT重みbf16 化により厳密には upstream と完全一致しない
  （実用上は同等想定だが、本番採用時は元実装との出力比較を推奨）。
- **7B（`configs_7b` / `models/dit`）は未検証**（attention のみ同様パッチ済、norm config は未変更）。
