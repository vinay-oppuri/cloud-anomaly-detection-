from __future__ import annotations

import random
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm.auto import tqdm


TensorDType = Literal["float", "long"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    dataset: Dataset[tuple[torch.Tensor, torch.Tensor]],
    *,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler[int] | None = None,
    num_workers: int = 0,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    effective_shuffle = shuffle if sampler is None else False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=effective_shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    input_dtype: TensorDType,
    progress_desc: str | None = None,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    iterator = loader
    progress_bar: tqdm | None = None
    if progress_desc:
        progress_bar = tqdm(
            loader,
            desc=progress_desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = progress_bar

    for batch_x, batch_y in iterator:
        features = _prepare_features(batch_x, device=device, dtype=input_dtype)
        labels = batch_y.to(device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        if progress_bar is not None:
            avg_loss = total_loss / max(1, total_items)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    if progress_bar is not None:
        progress_bar.close()

    return total_loss / max(1, total_items)


def evaluate_model(
    model: nn.Module,
    loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: torch.device,
    input_dtype: TensorDType,
    progress_desc: str | None = None,
) -> tuple[float, torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_items = 0

    logits_list: list[torch.Tensor] = []
    labels_list: list[torch.Tensor] = []

    iterator = loader
    progress_bar: tqdm | None = None
    if progress_desc:
        progress_bar = tqdm(
            loader,
            desc=progress_desc,
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        iterator = progress_bar

    with torch.inference_mode():
        for batch_x, batch_y in iterator:
            features = _prepare_features(batch_x, device=device, dtype=input_dtype)
            labels = batch_y.to(device=device, dtype=torch.long)
            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = int(labels.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

            logits_list.append(logits.detach().cpu())
            labels_list.append(labels.detach().cpu())
            if progress_bar is not None:
                avg_loss = total_loss / max(1, total_items)
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    if progress_bar is not None:
        progress_bar.close()

    avg_loss = total_loss / max(1, total_items)
    all_logits = torch.cat(logits_list, dim=0) if logits_list else torch.empty((0, 0))
    all_labels = torch.cat(labels_list, dim=0) if labels_list else torch.empty((0,), dtype=torch.long)
    return avg_loss, all_logits, all_labels


def _prepare_features(batch_x: torch.Tensor, *, device: torch.device, dtype: TensorDType) -> torch.Tensor:
    if dtype == "long":
        return batch_x.to(device=device, dtype=torch.long, non_blocking=True)
    return batch_x.to(device=device, dtype=torch.float32, non_blocking=True)

