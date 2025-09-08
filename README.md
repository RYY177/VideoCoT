# VideoCoT
Long video understanding pipeline

link: https://o1ddfyjva23.feishu.cn/wiki/Kqclwp0RaiFwGokjJJ2cC9c0nQh?from=from_copylink

---
### Update: 2025.09.08
In the split.py function, Step 1 and Step 2.1 have been completed. Scene segmentation was implemented using `scenedetect`, along with frame sampling for each scene. 

The scene length and number of samples can be flexibly adjusted according to the input token limits of the LLM. The sampling methods include both uniform sampling and dynamic sampling.
