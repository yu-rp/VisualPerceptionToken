# Qwen VPT Model
Our modifications to the *transformers* library include adding a new file, `modeling_qwen2_vl_vpt.py`.  

Qwen+VPT continues to use the tokenizer from Qwen2-VL.  

The processor for Qwen+VPT extends the original Qwen2-VL processing pipeline by incorporating processors from DINOv2 and SAM.  

The configuration for Qwen+VPT builds upon the original Qwen2-VL configuration with additional parameters:  

1. **Storing Visual Re-Encoding Token IDs and Image Placeholder Token IDs in the Model**  
   - `detection_image_token_id = None`  or `int`
   - `detection_action_id = None`    or `int`
   - `detection_action_start_id = None`    or `int`
   - `clip_image_token_id = None`    or `int`
   - `clip_action_id = None`    or `int`
   - `clip_action_start_id = None`    or `int`
   - `seg_image_token_id = None`    or `int`
   - `seg_action_id = None`    or `int`
   - `seg_action_start_id = None`    or `int`

2. **Controlling the Use of Vision Re-Encoding Tokens**  
   - `num_inner_forward_run = 2` (enabled) or `1` (disabled)  

3. **Selecting the Type of Vision Re-Encoding Tokens**  
   - This allows the model to integrate the corresponding additional vision encoder.  
   - `vision_encoder_ls = [dino], [sam], [clip]` or `[]`  

4. **Configuring Mask Modeling Behavior**  
   - `para_mask_id = 1` (enabled) or `0` (disabled)  
   - `para_mask_ratio = 0.5` (controls the proportion of masked samples)  

`VPT_Qwen2VLForConditionalGeneration` is the Qwen+VPT model, which inherits from `Qwen2VLForConditionalGeneration`. The primary difference between the Qwen+VPT model and the original model in the forward process lies in the use of Vision Re-Encoding tokens. To accommodate this, the model performs two forward passes:  

- The first forward pass generates the hidden states for the Re-Encoding control tokens.  
- The second forward pass produces the final output.  

In the code, variables named with `*_first_round` correspond to the first forward pass.

Instead of controlling the attention mask, we implement mask modeling by directly masking out the question and image tokens after the first forward pass, preventing them from being used in the second pass.

# Changes to LLaMA-Factory
The primary modification to *LLaMA-Factory* enables its data processing pipeline to distinguish which vision encoder should be used for each input image and apply the corresponding processing. Our dataset currently includes four types of images:  

1. **Original Image List** – Directly processed by the original vision encoder.  
2. **Image List Processed by DINOv2** – Triggered by Vision Re-Encoding Token (DINO).  
3. **Image List Processed by SAM** – Triggered by Vision Re-Encoding Token (Seg).  
4. **Image List Re-Encoded by the original vision encoder in the second round** – Triggered by Vision Re-Encoding Token (CLIP).  

Most of our modifications to the `LLaMA-Factory/src/llamafactory/data` directory are aimed at supporting the processing of image types 2–4.  

Additionally, in the `_encode_supervised_example` function of `LLaMA-Factory/src/llamafactory/data/processors/supervised.py`, we mask out the labels for the Re-Encoding Control Tokens.  

The remaining modifications to *LLaMA-Factory* focus on ensuring correct configuration and initialization of the `VPT_Qwen2VLForConditionalGeneration` model. These changes are primarily located in the `LLaMA-Factory/src/llamafactory/train/sft` and `LLaMA-Factory/src/llamafactory/hparams` directories.

To facilitate better control over the model architecture, we load the pretrained model directly using its state dict instead of using *safetensors*. To rerun the entire tuning process, models stored in *safetensor* format on Hugging Face need to be manually saved locally in advance.

```python
from transformers import Qwen2VLForConditionalGeneration, AutoModel, SamModel
qwen2b = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
qwen7b = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
dino = AutoModel.from_pretrained('facebook/dinov2-large')
sam = SamModel.from_pretrained("facebook/sam-vit-large")
qwen2b.save_pretrained("/root/models/Qwen2-VL-2B-Instruct", safe_serialization = False)
qwen7b.save_pretrained("/root/models/Qwen2-VL-7B-Instruct", safe_serialization = False)
dino.save_pretrained("/root/models/DINOv2-Large", safe_serialization = False)
sam.save_pretrained("/root/models/Sam-Large", safe_serialization = False)
```