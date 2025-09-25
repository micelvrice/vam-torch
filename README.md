# VAM: Value-Attention Merging for KV Cache Optimization in LLMs

This repository provides the **PyTorch implementation** of **VAM (Value-Attention Merging)**, a plug-in framework for **KV cache optimization** in large language models (LLMs).  
VAM dynamically updates the value vectors in the KV cache during autoregressive inference by merging them with attention outputs, addressing the limitations of static KV caches.  
It can be seamlessly integrated into existing cache compression methods such as H2O, PyramidKV, SnapKV, and ZipCache.

---

## 🔥 Features
- 🚀 **Plug-and-play**: Compatible with Hugging Face `transformers` and existing KV cache compression algorithms.  
- 📉 **Memory-efficient**: Reduces cache redundancy while preserving model accuracy.  
- 📈 **Performance gain**: Demonstrated improvements on long-context benchmarks (e.g., LongBench, synthetic tasks).  

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/VAM.git
cd VAM
pip install -r requirements.txt
