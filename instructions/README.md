
### Slime Env Setup

```bash
# cuda 12.9 (nvcc -V, nvidia-smi)
cd OpenClaw-RL
conda create --name openclaw-rl python=3.12 -y
conda activate openclaw-rl
 
pip install \
  torch==2.9.1+cu129 \
  torchvision==0.24.1+cu129 \
  torchaudio==2.9.1+cu129 \
  --index-url https://download.pytorch.org/whl/cu129
 
pip install -r requirements.txt

# DeepEP
git clone https://github.com/deepseek-ai/DeepEP.git
cd DeepEP
pip install -e . --no-build-isolation
cd ..

pip install -e slime/slime/backends/megatron_utils/kernels/int4_qat --no-build-isolation
 
# apex
git clone https://github.com/NVIDIA/apex.git
cd apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 pip install -v --no-build-isolation .
cd ..

# flash_attn
export MAX_JOBS=8
pip install --no-build-isolation -v flash-attn==2.7.4.post1
 
# flashinfer
pip install "flashinfer-jit-cache==0.6.3" --index-url https://flashinfer.ai/whl/cu129

# megatron-bridge
pip install "megatron-bridge @ git+https://github.com/fzyzcjy/Megatron-Bridge.git@35b4ebfc486fb15dcc0273ceea804c3606be948a" --no-build-isolation

# TransformerEngine
export NVTE_FRAMEWORK=pytorch
pip install --no-build-isolation "transformer_engine[pytorch,core_cu12]==2.10.0"

# apt
apt-get update
apt-get install -y python3-apt
```

If you want to use Qwen3.5 (omit this if only use Qwen3)

```
# upgrade transformers
pip install transformers==5.3.0
```

If you want to use Qwen3.5 for multimodal (omit if only use Qwen3, or use Qwen3.5 text only)

```
# Megatron-Bridge for qwen35 vl
cd OpenClaw-RL
git clone --recursive https://github.com/NVIDIA-NeMo/Megatron-Bridge.git Megatron-Bridge-qwen35
cd Megatron-Bridge-qwen35
git checkout ebca893607d48388a6c083bfc143bc05621cc753
git submodule update --init --recursive

# Megatron-LM comes from the Megatron-Bridge submodule
cd 3rdparty/Megatron-LM
git checkout 17a67b9a97fb11a75933fd7f76ad76e1ac98a53d
cd /path/to/Megatron-Bridge-qwen35

python3 -m pip uninstall -y megatron-bridge megatron-core mbridge
python3 -m pip install --no-deps -e ./3rdparty/Megatron-LM
python3 -m pip install --no-deps -e ./
```

















