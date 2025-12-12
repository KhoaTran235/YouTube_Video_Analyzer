import onnxruntime as ort
from transformers import AutoTokenizer
import numpy as np
import torch


print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())

if "CUDAExecutionProvider" in ort.get_available_providers():
    provider = "CUDAExecutionProvider"
else:
    provider = "CPUExecutionProvider"


session = ort.InferenceSession("onnx_lora_bert/model.onnx", providers=[provider])
print("Using provider:", session.get_providers())

tokenizer = AutoTokenizer.from_pretrained("onnx_lora_bert")
texts = ["This movie was fantastic!", "I did not like the film."]
inputs = tokenizer(texts, return_tensors="np", padding=True, truncation=True)

outputs = session.run(
    None,
    {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "token_type_ids": inputs["token_type_ids"],
    }
)
print(outputs)
logits = outputs[0]
predicted_class_ids = np.argmax(logits, axis=-1)  # mảng [2, 0] chẳng hạn
print(predicted_class_ids)
# In kết quả
for text, pred_id in zip(texts, predicted_class_ids):
    print(f"Text: {text}\nPredicted class: {pred_id}\n")