import evaluate

class MetricEvaluator:

    def __init__(self, cache_dir):
        self.metric = evaluate.load("accuracy", cache_dir=cache_dir)

    def __call__(self, eval_preds):
        return self.compute_metrics(eval_preds)

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return self.metric.compute(predictions=preds, references=labels)


def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        # Depending on the model and config, logits may contain extra tensors,
        # like past_key_values, but logits always come first
        logits = logits[0]
    return logits.argmax(dim=-1)