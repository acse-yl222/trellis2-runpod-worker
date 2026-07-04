# 部署指南 — Apple Silicon 专用路线(不用本地 Docker)

你在 M 芯片的 Mac 上,**不要本地 build**。用 RunPod 的 GitHub 集成,让 RunPod 在它自己的 amd64 机器上构建镜像。你只需要 push 代码。

---

## 一次性准备

1. 装 Git(Mac 一般自带,`git --version` 能出版本就行)。
2. 有一个 GitHub 账号。

---

## 第 1 步:把这个文件夹变成 GitHub 仓库

在这个 `trellis2-runpod-worker` 文件夹里(终端 `cd` 进来),依次跑:

```bash
git init
git add .
git commit -m "TRELLIS.2 runpod worker"
```

然后在 GitHub 网页上 New repository(设为 Private 也行),建好后按它给的提示:

```bash
git remote add origin https://github.com/<你的用户名>/trellis2-runpod-worker.git
git branch -M main
git push -u origin main
```

## 第 2 步:在 RunPod 连接 GitHub 并创建 Endpoint

1. 打开 https://console.runpod.io/serverless → **New Endpoint**。
2. 选 **GitHub**(不是 Docker Registry)。第一次会让你 **授权 RunPod 访问你的 GitHub**,授权后选中 `trellis2-runpod-worker` 仓库、`main` 分支。
3. RunPod 会自动发现根目录的 `Dockerfile`。
4. **GPU 选择**:主力选 **48GB 档(L40S / RTX 6000 Ada / A40 / A6000)**。要跑 1536³ 且在意速度可加 80GB(H100 / A100)。
   - ⚠️ **不要勾 RTX 5090 / B200**(Blackwell),torch 2.6/cu124 不支持,镜像也没编译对应架构,选了会崩。
5. **Workers**:开发期 `Min 0 / Max 1`。(Min 0 = 不调用不收费,正是你要的。)
6. **Execution Timeout** ≥ `900` 秒。
7. **Container Disk** ≥ `40` GB。

## 第 3 步(推荐):加 Network Volume 存权重

当前 Dockerfile 默认 **不把权重烘焙进镜像**(这样 RunPod 的服务器构建更快、更不容易超限)。所以第一次冷启动时权重(约十几 GB)要下载。用一个 Network Volume 把它持久化,只下一次:

1. RunPod 左侧 **Storage → New Network Volume**,建一个(比如 50GB),记住它绑定的 region。
2. 创建 Endpoint 时挂载这个 volume(默认挂到 `/runpod-volume`)。
3. 在 Endpoint 的 **Environment Variables** 里加一条:
   ```
   HF_HOME = /runpod-volume/hf-cache
   ```
   之后权重就下载并常驻在 volume 里,后续冷启动只剩加载时间。

> 注意:volume 绑定 region 会限制可选 GPU 池;而且 volume 按 GB/月持续计费(哪怕不调用)。如果你更想要"闲置绝对零成本",就跳过 volume,改成在本地或 GitHub 上把 `BAKE_WEIGHTS=1` 传给构建(但那样镜像 25–35GB,GitHub 服务器构建会更久)。对间歇调用,volume 那点存储费通常可忽略,我建议先用 volume。

## 第 4 步(推荐):配 S3 / R2 输出,避免大 GLB 撑爆响应

大 GLB 走 base64 会超响应体上限。配一个对象存储(Cloudflare R2 免费额度够用),在 Environment Variables 加:

```
BUCKET_ENDPOINT_URL     = https://<accountid>.r2.cloudflarestorage.com
BUCKET_ACCESS_KEY_ID    = ...
BUCKET_SECRET_ACCESS_KEY= ...
```

配了之后 worker 返回 `glb_url`(预签名链接);没配则返回 `glb_base64`(超过 14MB 会报错提示你配桶)。

## 第 5 步:Deploy → 首次构建

点 Deploy。**首次构建会比较久**(要编译 CuMesh / FlexGEMM / nvdiffrec 的 CUDA 扩展),之后 RunPod 有层缓存会快很多。构建日志在 Endpoint 页面能实时看到——如果哪个扩展编译失败,把日志发我,我按报错改 Dockerfile。

## 第 6 步:调用

```bash
export RUNPOD_API_KEY=rpa_xxx   # console → Settings → API Keys

python client.py --endpoint-id <ENDPOINT_ID> \
  --image-url https://raw.githubusercontent.com/microsoft/TRELLIS.2/main/assets/example_image/T.png \
  --resolution 1024 --mode async -o out.glb
```

或者直接 curl:

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"image_url":"https://raw.githubusercontent.com/microsoft/TRELLIS.2/main/assets/example_image/T.png","resolution":"1024"}}'
```

---

## 输入参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `image_url` / `image_base64` | 必填其一 | 输入图,大图走 URL |
| `resolution` | `"1024"` | `"512"` / `"1024"` / `"1536"` → 官方 512 / 1024_cascade / 1536_cascade |
| `seed` | `0` | 随机种子 |
| `preprocess` | `true` | 抠图/裁剪;已有 alpha 蒙版可设 false |
| `decimation_target` | `500000` | GLB 减面目标(10万–100万) |
| `texture_size` | `2048` | 1024 / 2048 / 4096 |
| `ss_sampler` / `shape_sampler` / `tex_sampler` | 官方默认 | 覆写 steps / guidance_strength / guidance_rescale / rescale_t |

三阶段采样器的**真实官方默认值**(已写进 handler,来自 app.py):
- ss(稀疏结构):steps 12, guidance 7.5, rescale 0.7, rescale_t 5.0
- shape(形状):steps 12, guidance 7.5, rescale 0.5, rescale_t 3.0
- tex(材质):steps 12, guidance 1.0, rescale 0.0, rescale_t 3.0

## 已知的坑

- **首次请求偏慢**:nvdiffrast 运行时 JIT 编译 kernel,每个全新 worker 的第一单多花一两分钟,之后缓存。属正常。
- **权重下载**:未配 Network Volume 时,第一次冷启动会下载十几 GB 权重。配了 `HF_HOME=/runpod-volume/hf-cache` 就只下一次。
- **依赖以 setup.sh 为准**:handler/Dockerfile 严格照官方 setup.sh 和 app.py 写。若上游改了 `pipeline.run` 签名或加了新依赖,构建/运行会报错,把日志发我同步即可。
- **许可证**:TRELLIS.2 本体 MIT,但 GLB 烘焙链路里的 nvdiffrast / nvdiffrec 有各自独立许可证,商用前单独确认。

## 更新镜像

改了代码后,只要 `git push`,RunPod 会自动重新构建并滚动更新 endpoint——不用手动 rebuild。
