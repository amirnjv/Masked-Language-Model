from transformers import pipeline

pred_model = pipeline("fill-mask", model = "MLP_TrainedModels")

text = "Paris is the [MASK] of France."

preds = pred_model(text)

for pred in preds:
    print(f">>> {pred['sequence']}")
    