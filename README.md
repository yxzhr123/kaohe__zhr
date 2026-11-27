# 拼图求解器（YOLO + OpenCV）

自动求解 [PlayJigsaw](https://puzzle.moyunliuyin.me/?mode=exam) 考试模式：**25 格 = 20 张照片 + 5 个纯白块**（5×5，不旋转）。

单文件实现，`puzzle_solver.py` 即全部代码。

## 实测

```
25/25  —— 照片 20/20 + 白块 5/5，步数 35，用时约 2 分 15 秒
24/25  —— 照片 20/20 + 白块 4/5
```

## 下载

| 内容 | 位置 | 大小 |
|---|---|---|
| 数据集（2800 张 + 标签） | [Release 附件 dataset.zip](https://github.com/yxzhr123/kaohe__zhr/releases/download/v1.0/dataset.zip) | 524 MB |
| 25/25 实跑录屏 | [Release 附件 demo.mp4](https://github.com/yxzhr123/kaohe__zhr/releases/download/v1.0/puzzle_demo_25of25.mp4) | 102 MB |

数据集解压后放到仓库根目录的 dataset/ 下即可用 dataset/data.yaml 训练。

## 演示

录屏里一局完整流程：照片 20 格（约 72 秒，几乎全部一次到位）→ 白块 5 格（约 42 秒）
→ 25/25，步数 35，用时 2 分 15 秒。

## 仓库结构

```
puzzle_solver.py          主程序（单文件，1720 行）
weights/best.pt           YOLO11s 权重（15 类，18.4 MB）
dataset/
  data.yaml               数据集配置（相对路径）
  images/train  2400 张   训练图
  images/val     400 张   验证图
  labels/train  2400 个   YOLO 格式标签
  labels/val     400 个
requirements.txt
```

## 安装

```bash
pip install -r requirements.txt
python -m playwright install chrome
```

## 运行

```bash
# 1. 先在浏览器里手动选 25 格并点开始
# 2. 然后运行：
python puzzle_solver.py --games 1 --minutes 12
```

日志写入 `final25.log`，结束后截图 `final_end.png`。

权重默认读 `weights/best.pt`（相对脚本位置），也可用环境变量覆盖：

```bash
set YOLO_WTS=D:\path\to\best.pt
```

## 模型

| 项 | 值 |
|---|---|
| 架构 | YOLO11s（ultralytics） |
| 训练轮数 | 60 epoch，batch 4，imgsz 1280 |
| 类别数 | 15 |
| 指标（第 60 轮） | precision 0.989 / recall 0.969 / mAP50 0.988 / mAP50-95 0.968 |

类别：`dog cat elephant desktop_computer bicycle car chair cup airplane backpack umbrella suitcase penguin tafei nailong`

### 重新训练

```bash
yolo detect train model=yolo11s.pt data=dataset/data.yaml epochs=60 imgsz=1280 batch=4
```

### 模型在程序中的两处用途

1. **参考图 20 格分类**：整张参考图推理一次，按检测框中心落格
2. **碎片类别核对**：滑窗命中后取该位置碎片的 YOLO 类别与参考格类别比对，
   一致则 `分数 = 几何分×0.8 + 0.2`（日志里的 `同类+` / `异类`）

## 算法

### 阶段一：照片块（20 格）

| 步骤 | 做法 |
|---|---|
| 模板 | 参考格**只取中心 76%**，去掉碎片白边/凸起的干扰 |
| 定位 | 5 个缩放档（0.92~1.08）全画布滑窗，每格取前 2 峰 |
| 亮度惩罚 | 候选区域亮度必须接近参考格，否则降分（区分白杯子/黑杯子） |
| 白底特化 | 白色占比 >45% 的格子额外做 Canny 边缘匹配，取两者较大值 |
| 类别融合 | YOLO 判定同类 → `分数×0.8 + 0.2` |
| 全局分配 | `scipy.optimize.linear_sum_assignment`——一块碎片只给一个格子 |
| 过滤 | 剔除棋盘内的峰值（已放好的同类块） |
| 门限 | <0.60 不拖；0.60~0.75 只有明确"异类"才拒；空类别放行（透明雨伞） |
| 顺序 | 四角 → 四边 → 内圈，内部格必须有已放好的邻居；角/边格由棋盘边界当锚点 |
| 拖拽 | 起点用 `cv2.moments` 质心，落点格子中心，慢速分步移动 |
| 校验 | **只看游戏进度计数是否 +1**；失败微调 ±8，仍失败则拖回原位 |

### 阶段二：白块（5 格，照片 20/20 锁定后才开始）

| 步骤 | 做法 |
|---|---|
| 提取 | HSV 阈值 + 连通域，**不做腐蚀**（凸起只有 4~6px，腐蚀会抹掉） |
| 粘连拆分 | 面积 ≈ k 倍标准块时按宽度/高度切成 k 块 |
| 四边判定 | `get_edge_features`：取每条边中段 64% 的逐行外沿坐标，与中位数比较——多数行更靠外 = Tab，更靠内 = Blank，否则 Flat；`approxPolyDP` 平滑 + 凸包兜底 |
| 空位需求 | 由邻居块的四边**取反**推导：邻居 Tab→本块 Blank、Blank→Tab、Flat→Flat、棋盘边界→Flat |
| 匹配 | 四边全吻合 → 平边数量相同 → 无平边需求时用全吻合那批 |
| 消元收尾 | 剩 1~3 坑时逐格重检白块，一块一块试，进度为准 |

## 关键实现说明

**为什么模板只取中心 76%**
整张参考格做模板时，碎片的白边与凸起会把正确块的相关性拉到 0.7 左右，和错块的 0.7 混在一起。
只取中心区域后，正确块稳定 0.84~0.99，错块掉到 0.6 以下。

**为什么白块不能腐蚀**
为了分开粘连的白块一开始用了 `erode(7×7)`，结果凸起被腐蚀掉，面积比只剩 0.47，
四条边永远判成 Flat——形状信息被自己抹掉了。

**为什么用进度计数当判据**
像素差只能说明"有东西放进去"，不能说明"放对了"：错块放上去像素差同样很大。
游戏进度只对正确的块 +1，是唯一可信的判据。

## 已知限制

- 页面必须已经在 25 格局面；脚本自动切难度对自定义下拉无效
- 偶尔有 1 个碎片检测不到（被其它块压住或贴边），这一局照片会停在 19/20
- Chrome 被外部关闭会导致整局中断（日志里是 `TargetClosedError`）

## License

MIT
