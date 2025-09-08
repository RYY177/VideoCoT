
import os
import math
import cv2
import numpy as np
from typing import List, Tuple, Optional
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector, AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg



def detect_scenes(video_path, threshold=35.0, min_scene_sec=0, use_adaptive=False):
    video = open_video(video_path)
    fps = video.frame_rate or 30.0
    scene_manager = SceneManager()

    if use_adaptive:
        # 两阶段自适应，运动多时更稳；
        scene_manager.add_detector(AdaptiveDetector())
    else:
        scene_manager.add_detector(
            ContentDetector(
                threshold=threshold,
                min_scene_len=int(min_scene_sec * fps) if min_scene_sec > 0 else 0
            )
        )

    scene_manager.detect_scenes(video, show_progress=True)
    return scene_manager.get_scene_list()

def merge_short_scenes(scene_list, min_scene_sec):
    def dur(s): return s[1].get_seconds() - s[0].get_seconds()
    changed = True
    while changed and len(scene_list) > 1:
        changed = False
        for i, sc in enumerate(scene_list):
            if dur(sc) < min_scene_sec:
                # 与更短的一侧邻居合并
                if i == 0:
                    j = 1
                elif i == len(scene_list) - 1:
                    j = i - 1
                else:
                    left, right = scene_list[i - 1], scene_list[i + 1]
                    j = i - 1 if dur(left) <= dur(right) else i + 1
                s, e = min(i, j), max(i, j)
                merged = (scene_list[s][0], scene_list[e][1])
                scene_list = scene_list[:s] + [merged] + scene_list[e + 1:]
                changed = True
                break
    return scene_list

def cap_to_n_scenes(scene_list, target_n):
    def dur(s): return s[1].get_seconds() - s[0].get_seconds()
    while len(scene_list) > target_n and len(scene_list) > 1:
        i_min = min(range(len(scene_list)), key=lambda i: dur(scene_list[i]))
        if i_min == 0:
            j = 1
        elif i_min == len(scene_list) - 1:
            j = i_min - 1
        else:
            j = i_min - 1 if dur(scene_list[i_min - 1]) <= dur(scene_list[i_min + 1]) else i_min + 1
        s, e = min(i_min, j), max(i_min, j)
        merged = (scene_list[s][0], scene_list[e][1])
        scene_list = scene_list[:s] + [merged] + scene_list[e + 1:]
    return scene_list

def split_video_semantic(video_path, target_n=10,
                         threshold=35.0,
                         detect_min_scene_sec=0,   # 检测阶段尽量小/0
                         final_min_scene_sec=10,   # 合并阶段再约束最短时长
                         use_adaptive=False,
                         show_progress=True):
    # 1) 检测
    scenes = detect_scenes(video_path, threshold=threshold,
                           min_scene_sec=detect_min_scene_sec,
                           use_adaptive=use_adaptive)
    # 2) 合并到满足最终最短时长
    scenes = merge_short_scenes(scenes, final_min_scene_sec)
    # 3) 进一步合并到 N
    # scenes = cap_to_n_scenes(scenes, target_n)
    # 4) 切视频
    # split_video_ffmpeg(video_path, scenes, show_progress=show_progress)
    return scenes



# --------- 工具函数：时间码/帧号 ---------

def tc_to_frame(tc, fps: float) -> int:
    # tc 可为 scenedetect.FrameTimecode 或秒数（float）
    if hasattr(tc, "frame_num"):
        return int(tc.frame_num)
    if hasattr(tc, "get_seconds"):
        return int(round(tc.get_seconds() * fps))
    # float 秒数
    return int(round(float(tc) * fps))

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# --------- 采样策略：均匀采样 ---------

def uniform_sample_indices(start_f: int, end_f: int, *,
                           frames_per_scene: Optional[int] = None,
                           fps_samples: Optional[float] = None,
                           video_fps: float = 30.0) -> List[int]:
    """
    - frames_per_scene: 每段固定抽多少张
    - fps_samples: 每秒抽多少张（比如 1.0 表示 1 fps）
    """
    length = max(0, end_f - start_f + 1)
    if length <= 0:
        return []

    if frames_per_scene is not None and frames_per_scene > 0:
        count = min(frames_per_scene, length)
        # 在区间内均匀取样，避免端点贴边
        return [start_f + int(round((i + 0.5) * length / count)) for i in range(count)]

    if fps_samples is not None and fps_samples > 0:
        step = max(1, int(round(video_fps / fps_samples)))
        return list(range(start_f, end_f + 1, step))

    # 默认：每段取一张中间帧
    return [start_f + length // 2]

# --------- 采样策略：运动驱动（代表帧） ---------

def motion_scores_in_scene(cap: cv2.VideoCapture, start_f: int, end_f: int,
                           sample_stride: int = 1) -> Tuple[List[int], List[float]]:
    """
    对 [start_f, end_f] 范围内按 stride 抽样，计算帧间差异作为运动分数：
    - 使用灰度 + 轻度高斯滤波，分数 = mean(abs(curr - prev)) / 255
    返回抽样到的帧号数组和对应分数（第一个样本分数为 0）。
    """
    indices, scores = [], []
    prev = None

    for f in range(start_f, end_f + 1, max(1, sample_stride)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (5, 5), 0)

        if prev is None:
            score = 0.0
        else:
            diff = cv2.absdiff(g, prev)
            score = float(np.mean(diff)) / 255.0  # 0~1
        prev = g
        indices.append(f)
        scores.append(score)

    if len(scores) > 0:
        scores[0] = 0.0  # 第一帧无前帧可比，记 0
    return indices, scores

def select_topk_with_nms(indices: List[int], scores: List[float],
                         k: int, min_gap_frames: int) -> List[int]:
    """
    选分数最高的 K 个帧，带最小间隔的“非极大值抑制”（NMS）。
    """
    if not indices:
        return []
    order = np.argsort(scores)[::-1]  # 从高到低
    selected = []
    used = np.zeros(len(indices), dtype=bool)

    for oi in order:
        if used[oi]:
            continue
        f = indices[oi]
        # 与已选帧保持最小间隔
        if any(abs(f - s) < min_gap_frames for s in selected):
            continue
        selected.append(f)
        # 可选：把附近一定范围标记掉，减少相邻重复
        if len(selected) >= k:
            break
    # 排序输出
    return sorted(selected)

# --------- 主函数：按场景抽帧 ---------

def extract_frames_per_scene(
    video_path: str,
    scene_list: List[Tuple[object, object]],  # [(start_tc, end_tc), ...]
    output_dir: str,
    strategy: str = "uniform",                # "uniform" or "motion"
    # frames_per_scene: int = 5,                # 均匀/运动模式都支持固定 K
    frames_per_scene: Optional[int] = None,   # 均匀/运动模式都支持固定 K
    fps_samples: Optional[float] = None,      # 均匀采样可用：按 fps 抽帧
    motion_stride: int = 2,                   # 运动分计算抽样步长（越大越快越粗糙）
    motion_min_gap_sec: float = 1.0,          # 运动模式下，代表帧之间的最小间隔
    jpeg_quality: int = 95,
    resize_shorter: Optional[int] = None,     # 可选：把短边缩放到该值，加速
) -> None:
    """
    将每个场景抽取代表帧并保存到 output_dir
    """
    ensure_dir(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    for si, (s_tc, e_tc) in enumerate(scene_list):
        start_f = tc_to_frame(s_tc, video_fps)
        end_f   = tc_to_frame(e_tc, video_fps)
        if end_f < start_f:
            continue

        scene_dir = os.path.join(output_dir, f"scene_{si:03d}")
        ensure_dir(scene_dir)

        if strategy == "uniform":
            frame_ids = uniform_sample_indices(
                start_f, end_f,
                frames_per_scene=frames_per_scene,
                fps_samples=fps_samples,
                video_fps=video_fps
            )

        elif strategy == "motion":
            # 1) 计算运动分
            indices, scores = motion_scores_in_scene(
                cap, start_f, end_f, sample_stride=motion_stride
            )
            if len(indices) == 0:
                continue
            # 2) NMS 选 K 张，保证时间间隔
            min_gap_frames = int(round(motion_min_gap_sec * video_fps))
            frame_ids = select_topk_with_nms(indices, scores,
                                             k=frames_per_scene,
                                             min_gap_frames=min_gap_frames)
            # 如果有效帧不足 K，再均匀补齐
            if len(frame_ids) < frames_per_scene:
                extra = uniform_sample_indices(
                    start_f, end_f,
                    frames_per_scene=frames_per_scene - len(frame_ids),
                    video_fps=video_fps
                )
                frame_ids = sorted(set(frame_ids + extra))

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        # 逐帧导出
        for rank, f in enumerate(frame_ids):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, frame = cap.read()
            if not ok:
                continue

            if resize_shorter is not None and resize_shorter > 0:
                h, w = frame.shape[:2]
                short = min(h, w)
                scale = resize_shorter / short
                nh, nw = int(round(h * scale)), int(round(w * scale))
                frame = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)

            t_sec = f / video_fps
            out_name = os.path.join(scene_dir, f"f{f:08d}_t{t_sec:07.2f}_{rank:02d}.jpg")
            cv2.imwrite(out_name, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    cap.release()
    print(f"[done] frames saved to: {output_dir}")


if __name__ == "__main__":
    video_path = '/code/yangyang.ren/MLVU/mvlu_data/MLVU/video/7_topic_reasoning/movie101_21.mp4'
    # video_path = '/code/yangyang.ren/MLVU/mvlu_data/MLVU/video/7_topic_reasoning/9.mp4'

    # 1) Step1：Scene Segmentation 
    scenes = split_video_semantic(video_path, target_n=10, threshold=35.0, detect_min_scene_sec=4, final_min_scene_sec=20)
    for i in range(len(scenes)):
        print(scenes[i])
    print(len(scenes))



    # 2) Step 2.1: Frame Sampling
    # Uniform sampling: extract 1 fps per scene (number of frames varies depending on scene length)
    extract_frames_per_scene(
        video_path, scenes, "./split/frames_fps",
        strategy="uniform", fps_samples=1.0
    )

    # # Uniform sampling: take 10 frames per scene (not equally divided by duration)
    # extract_frames_per_scene(
    #     video_path, scenes, "./frames_uniform",
    #     strategy="uniform", frames_per_scene=10
    # )

    # # Motion-based sampling: select 10 representative frames per scene with a minimum interval of 1.2 seconds
    # extract_frames_per_scene(
    #     video_path, scenes, "./frames_motion",
    #     strategy="motion", frames_per_scene=10,
    #     motion_stride=2, motion_min_gap_sec=1.2,
    #     resize_shorter=720  # Optional: resize shorter side to 720 to speed up
    # )
