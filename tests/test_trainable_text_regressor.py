import json

import torch
from torch import nn

from kswing_sentinel.training import TrainingPipeline


class TinyTextRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.05], dtype=torch.float32))
        self.bias = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))

    def forward_texts(self, texts: list[str]) -> torch.Tensor:
        lengths = torch.tensor([len(text.split()) for text in texts], dtype=torch.float32, device=self.weight.device)
        return lengths * self.weight + self.bias


def test_training_pipeline_can_train_text_regressor_and_save_artifacts(tmp_path):
    rows = [
        {"text": "strong demand recovery", "target": 1.4},
        {"text": "earnings guidance raised", "target": 1.5},
        {"text": "supply disruption hurts margin", "target": 2.1},
        {"text": "order backlog improved", "target": 1.3},
    ]
    model = TinyTextRegressor()
    initial_weight = float(model.weight.detach().cpu())

    weights_path, artifact_path, metrics_path = TrainingPipeline().train_text_regressor(
        rows=rows,
        text_key="text",
        label_key="target",
        artifact_dir=tmp_path,
        model=model,
        batch_size=2,
        epochs=5,
        lr=1e-2,
        device="cpu",
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert weights_path.exists()
    assert artifact["label_key"] == "target"
    assert artifact["text_key"] == "text"
    assert isinstance(metrics, list)
    assert len(metrics) == 5
    assert float(model.weight.detach().cpu()) != initial_weight
