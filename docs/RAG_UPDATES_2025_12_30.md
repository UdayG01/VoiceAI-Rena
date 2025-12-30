# Voice Agent TTS Optimization – December 30, 2025

**Author:** Uday Gupta  
**Commit ID:** `8b92aedb18967cfc0759520df243fe2dadbcdffe`  
**Description:** Explicitly offload TTS to CUDA

---

## Summary of Changes

Optimized the **Veena TTS** model loading mechanism to forcefully offload the model to the GPU using **CUDA**. This change prevents accidental CPU offloading or inefficient device mapping (e.g., across multiple GPUs or CPU/GPU split) that can occur with `device_map="auto"`.

This ensures that the entire Text-to-Speech generation pipeline runs on the GPU, significantly improving response latency and real-time performance.

---

## 1. Explicit TTS Model Placement

**File:** `src/test_veena.py`

* Modified the `AutoModelForCausalLM.from_pretrained` arguments for the Veena model.
* Changed `device_map` from `"auto"` (or implicit default) to a strict mapping:
  ```python
  device_map={"": torch.cuda.current_device()}
  ```
* This forces all model layers to reside on the currently active CUDA device (e.g., `cuda:0`), avoiding model parallelism overhead or CPU fallback.

---

## 2. Updated Model Loading Configuration

**File:** `src/test_veena.py`

* The configuration now looks as follows:
  ```python
  veena_model = AutoModelForCausalLM.from_pretrained(
      "maya-research/veena-tts",
      quantization_config=quantization_config,
      device_map={"": torch.cuda.current_device()}, 
      trust_remote_code=True,
  )
  ```
* This complements the existing 4-bit `BitsAndBytesConfig` implementation.

---

## Resulting Improvements

* **Performance:** guarantees GPU utilization for the heaviest part of the voice pipeline (TTS generation).
* **Reliability:** Prevents runtime warnings or errors related to device mismatches when moving tensors between CPU and GPU.
* **Latency:** Lowers the time-to-first-byte (TTFB) for audio response generation.
