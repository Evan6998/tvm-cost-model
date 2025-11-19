### Low-Level Kernel Cost Model

Harivallabha Rangarajan (hrangara), Yi Shi (yishi3), Evan Huang (yongfenh)

### 1\. Introduction

Optimizing GPU kernels such as matrix multiplication and convolution is essential for accelerating deep learning workloads, yet existing autotuning frameworks like TVM AutoTVM require costly on-device measurements. Learned cost models offer a faster alternative by predicting runtime, but current models often lack transferability, invariance, and interpretability. This project aims to develop a learned, transferable, and explainable cost model for GPU kernel performance prediction and integrate it into TVM’s MetaSchedule, enabling more efficient and interpretable autotuning.

### 2\. Problem

The project aims to design a learned cost model that accurately predicts the runtime of low-level GPU kernel programs (e.g., deep learning operators) while generalizing across workloads and minor code variations. Existing autotuning frameworks like TVM AutoTVM rely on expensive on-device measurements, whereas a learned model could replace these with fast predictions. The key challenges include achieving **transferability** across different operators and input sizes, ensuring **invariance** to semantically equivalent program transformations, maintaining high **accuracy** in runtime prediction, and providing **explainability** to interpret performance factors. The model will learn to map low-level program representations to execution time on NVIDIA GPUs, focusing on representative kernels such as matrix multiplication and convolution.

### 3\. Status quo

**Tensor compilers and learned cost models**   
The TVM family pioneered learning-based operator autotuning. **AutoTVM** framed kernel optimization as learning a cost model that ranks candidate schedules using gradient-boosted trees (XGBoost) over hand-crafted loop and IR features, with transfer learning across workloads to reduce measurements \[1\]. Building on this, **Ansor** (the TVM auto-scheduler) eliminated templates and used evolutionary search guided by a learned cost model to find high-performance programs across CPU/GPU targets \[2\]. TVM’s newer **MetaSchedule** unifies these approaches with a pluggable cost-model interface (XGBoost, MLP, random, or none), feature extraction from TIR, and a search policy integrated with a measurement database for warm starts \[3\].

**Richer models for cost prediction and transfer**  
Recent work improves accuracy and *transferability* of cost models across operators and devices. **TenSet** provides a large-scale dataset for program performance modeling and explores warm-start strategies for TVM-style cost models \[4\]. **Transfer-Tuning** and related studies highlight cross-feature and cross-hardware transfer learning for tensor program cost models \[5\]. **ATFormer** demonstrates that transformer-based architectures can generalize across hardware configurations \[6\].

**Program representations and graph neural networks**  
Graph-structured representations better capture program semantics. **ProGraML** constructs language-agnostic program graphs from compiler IRs (control, data, and call edges) and achieves strong results on compiler optimization and analysis tasks \[7\]. Similarly, **Inst2Vec** learns LLVM-IR embeddings for heterogeneous device mapping and performance prediction \[8\]. **Ithemal** shows neural models can predict CPU basic block throughput more accurately than analytical simulators \[9\]. At higher abstraction levels, **DNNPerf** applies GNNs over computation graphs to predict deep learning training performance \[10\].

**Analytical and heuristic baselines**  
Traditional analytical models such as the **Roofline model** \[11\] and the GPU CPI model by **Hong and Kim** \[12\] provide useful upper bounds but cannot capture the subtle schedule-sensitive and device-specific variations seen in modern kernels.

**Other autotuners and schedulers**  
Beyond TVM, **Halide’s** autoschedulers \[13, 14\] use learned or random-forest cost models to outperform human experts. **FlexTensor** blends heuristic exploration with learned cost modeling on heterogeneous systems \[15\]. **Tensor Comprehensions** offers a polyhedral JIT autotuner for generating CUDA kernels \[16\].

**Gaps motivating this project**  
(1) **Generalization across kernels and code variants.** Existing boosted-tree and token-sequence models fail to generalize beyond their training distributions \[1, 4, 6\].  
(2) **Limited explainability.** Production cost models rarely expose interpretable reasons behind predictions; post-hoc methods such as SHAP partially help \[17\].  
(3) **Operator-IR granularity.** Graph-based approaches mainly operate on neural-network graphs rather than low-level TIR/loop-nest structures \[10\].  
(4) **Integration potential.** TVM’s MetaSchedule provides an ideal pluggable interface for a new graph-based model that enhances transferability and explainability \[3\].

### 4\. Implementation plans

**Plan**  
Stage 1: Data generation: Use TVM MetaSchedule/scripts to batch-generate candidates for multiple kernels across diverse shapes; measure on a single NVIDIA GPU  
Stage 2: Model design: Explore program representations and architectures  
Stage 3: Integration: Implement the model as a TVM PyCostModel and plug it into MetaSchedule  
Stage 4: Evaluation: Compare predictive accuracy against TVM’s XGB baseline. Test across multiple kernels (e.g., GEMM, Conv, Depthwise, BMM) and across different GPUs to assess generalization

**Datasets**  
We are currently surveying public datasets for low‑level GPU kernel performance. If no suitable option meets our coverage or metadata needs, we will construct our own dataset using TVM MetaSchedule (generated kernels, measured runtimes on a single NVIDIA GPU)
