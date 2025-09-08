# VideoCoT
Long video understanding pipeline

pipeline: https://o1ddfyjva23.feishu.cn/wiki/Kqclwp0RaiFwGokjJJ2cC9c0nQh?from=from_copylink

---
### Update: 2025.09.08
In `split.py`, Step 1 and Step 2.1 have been completed. Scene segmentation was implemented using `scenedetect`, along with frame sampling for each scene. 

The scene length and number of samples can be flexibly adjusted according to the input token limits of the LLM. The sampling methods include both uniform sampling and dynamic sampling.

```
    # Uniform sampling: extract 1 fps per scene (number of frames varies depending on scene length)
    extract_frames_per_scene(
        video_path, scenes, "./split/frames_fps",
        strategy="uniform", fps_samples=1.0
    )
```
```
    # Uniform sampling: take 10 frames per scene (not equally divided by duration)
    extract_frames_per_scene(
        video_path, scenes, "./frames_uniform",
        strategy="uniform", frames_per_scene=10
    )
```
```
    # Motion-based sampling: select 10 representative frames per scene with a minimum interval of 1.2 seconds
    extract_frames_per_scene(
        video_path, scenes, "./frames_motion",
        strategy="motion", frames_per_scene=10,
        motion_stride=2, motion_min_gap_sec=1.2,
        resize_shorter=720  # Optional: resize shorter side to 720 to speed up
    )
```
