#!/usr/bin/env python3
"""
Multi-Model Action Recognition Comparison Report Generator

Evaluates multiple V-JEPA models on the EK100 validation set and generates:
- Animated GIFs of sampled videos
- Interactive HTML report comparing top-5 predictions across models
- Ground truth vs predicted labels for easy model comparison
"""

import os
import sys
import json
import warnings
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import io
import base64
import urllib.request

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from decord import cpu, VideoReader
from PIL import Image
import imageio
import importlib
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def configure_logging(log_path: str):
    """Configure logging to both console and file."""
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Clear existing handlers to prevent duplicate log entries when re-running
    if logger.hasHandlers():
        logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # Also capture warnings through logging
    logging.captureWarnings(True)


from src.datasets.video_dataset import make_videodataset
from evals.video_classification_frozen.utils import make_transforms
from evals.video_classification_frozen.eval import make_dataloader
from evals.video_classification_frozen.eval import load_pretrained
from evals.video_classification_frozen.models import init_module

# ============================================================================
# Configuration
# ============================================================================

MODELS_CONFIG = [
    {
        "name": "V-JEPA2 (baseline)",
        "checkpoint": "preliminary_experiments/EK100-vjepa-16f-4pfs/latest.pt",
        "config": "configs/eval/vitl/ek100-ar.yaml",
    },
    {
        "name": "High V-JEPA",
        "checkpoint": "preliminary_experiments/EK100-long-vjepa-16f/latest.pt",
        "config": "configs/eval/vitl/ek100-ar-high-vjepa.yaml",
    },
    {
        "name": "High V-JEPA (Same data as baseline)",
        "checkpoint": "preliminary_experiments/EK100-long-vjepa-16f_extended/latest.pt",
        "config": "configs/eval/vitl/ek100-ar-high-vjepa_extended.yaml",
    },
    {
        "name": "High V-JEPA (Same data + patches)",
        "checkpoint": "preliminary_experiments/EK100-long-vjepa-16f-16x16/latest.pt",
        "config": "configs/eval/vitl/ek100-ar-high-vjepa_extended_16x16.yaml",
    },
    {
        "name": "High V-JEPA (4fps frames)",
        "checkpoint": "preliminary_experiments/EK100-long-vjepa-16f/latest.pt",
        "config": "configs/eval/vitl/ek100-ar-high-vjepa_4fps.yaml",
    },
    {
        "name": "V-JEPA2 - Causal learning",
        "checkpoint": "preliminary_experiments/EK100-vjepa-16f-4pfs-future-prediction/latest.pt",
        "config": "configs/eval/vitl/ek100-ar-causal-backbone.yaml",
    },
    {
        "name": "High V-JEPA on V-JEPA2",
        "checkpoint": "preliminary_experiments/EK100-long-vjepa-16f-16x16_post_training/latest.pt",
        "config": "configs/eval/vitl/ek100-ar-high-vjepa-post-trained.yaml",
    },
    {
        "name": "High V-JEPA on causal (4fps)",
        "checkpoint": "preliminary_experiments/EK100-long-vjepa-16f-16x16_post_training_causal_base/latest.pt",
        "config": "configs/eval/vitl/ek100-ar-high-vjepa-post-trained_causal_base.yaml",
    },
]

VERB_CLASSES_URL = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_verb_classes.csv"
NOUN_CLASSES_URL = "https://raw.githubusercontent.com/epic-kitchens/epic-kitchens-100-annotations/master/EPIC_100_noun_classes_v2.csv"

# ============================================================================
# Helper Functions
# ============================================================================

def download_class_labels(url: str, cache_file: Optional[str] = None) -> Dict[int, str]:
    """Download class labels from Epic Kitchens GitHub."""
    if cache_file and os.path.exists(cache_file):
        logger.info(f"Loading cached labels from {cache_file}")
        return json.load(open(cache_file))
    
    logger.info(f"Downloading labels from {url}")
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
        lines = data.strip().split('\n')[1:]  # Skip header
        labels = {}
        for line in lines:
            parts = line.split(',', 2)
            if len(parts) >= 2:
                idx = int(parts[0])
                name = parts[1].strip().strip('"')
                labels[idx] = name
        
        if cache_file:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            json.dump(labels, open(cache_file, 'w'))
        
        return labels
    except Exception as e:
        logger.warning(f"Failed to download labels: {e}")
        return {}


def load_validation_csv(csv_path: str) -> pd.DataFrame:
    """Load validation CSV file."""
    df = pd.read_csv(csv_path, header=None, delimiter=" ", names=["video_path", "label"])
    # Parse multi-task labels (verb, noun)
    df[["verb_id", "noun_id"]] = df["label"].str.split(",", expand=True).astype(int)
    return df

def save_validation_csv(df: pd.DataFrame, csv_path: str) -> None:
    """Save a validation DataFrame back to the original CSV format."""
    out_df = df.copy()

    # Reconstruct the original label column
    out_df["label"] = (
        out_df["verb_id"].astype(str)
        + ","
        + out_df["noun_id"].astype(str)
    )

    # Keep only the original columns and save
    out_df[["video_path", "label"]].to_csv(
        csv_path,
        sep=" ",
        header=False,
        index=False,
    )

def sample_random_videos(df: pd.DataFrame, n_samples: int = 10, seed: int = 42) -> pd.DataFrame:
    """Sample n random videos from the validation set."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    return df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)


def load_config(config_path: str) -> Dict:
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_model_checkpoint(config: dict, encoder_emb_dim: int, device: str = "cuda:0") -> Tuple[torch.nn.Module, Dict]:
    """Load pretrained model checkpoint."""

    args_exp = config.get("experiment")
    args_classifier = args_exp.get("classifier")
    num_probe_blocks = args_classifier.get("num_probe_blocks", 1)
    num_heads = args_classifier.get("num_heads", 16)

    args_data = args_exp.get("data")
    num_classes = args_data.get("num_classes")

    # Classifiers folder
    pretrain_folder = config.get("folder", None)
    eval_tag = config.get("tag", None)
    folder = os.path.join(pretrain_folder, "video_classification_frozen/")
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    checkpoint_path = os.path.join(folder, "latest.pt")
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    
    # Initialize classifier from checkpoint
    classifiers = []
    if "classifiers" in checkpoint:
        from src.models.attentive_pooler import AttentiveClassifier
        
        for state_dict in checkpoint["classifiers"]:
            classifier = AttentiveClassifier(
                embed_dim=encoder_emb_dim,
                num_heads=num_heads,
                depth=num_probe_blocks,
                num_classes=num_classes,
            )
            state_dict = {
                k.replace("module.", "", 1): v
                for k, v in state_dict.items()
            }
            msg = classifier.load_state_dict(state_dict)
            logger.info(f"loaded pretrained classifier with msg: {msg}")
            classifier.train(mode=False)
            classifiers.append(classifier.to(device))
    
    return classifiers


def load_data_for_model(config: dict, data_path: str, batch_size: int):
    args_data = config['experiment'].get("data")
    dataset_type = args_data.get("dataset_type", "VideoDataset")
    resolution = args_data.get("resolution", 224)
    num_segments = args_data.get("num_segments", 1)
    frames_per_clip = args_data.get("frames_per_clip", 16)
    frame_step = args_data.get("frame_step", 4)
    duration = args_data.get("clip_duration", None)
    normalization = args_data.get("normalization", None)
    allow_variable_length = args_data.get("allow_variable_length", False)
    args_model = config.get('model_kwargs').get('pretrain_kwargs')

    # Make Video Transforms
    val_loader, _ = make_dataloader(
        dataset_type=dataset_type,
        root_path=data_path,
        img_size=resolution,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_segments=num_segments,
        eval_duration=duration,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=1,
        rank=0,
        training=False,
        num_workers=16,
        normalization=normalization,
        patch_size=args_model['encoder']['patch_size'],
        allow_variable_length=allow_variable_length,
        tubelet_size=args_model['encoder']['tubelet_size'],
        shuffle=False,
    )
    
    return val_loader

def load_video_frames(video_path: str, frames_per_clip: int = 16, frame_step: int = 1) -> np.ndarray:
    """Load video frames using Decord."""
    try:
        vr = VideoReader(video_path, num_threads=-1, ctx=cpu(0))
        
        if len(vr) == 0:
            logger.warning(f"Video is empty: {video_path}")
            return None
        
        # Calculate frame indices
        clip_len = frames_per_clip * frame_step
        if len(vr) < clip_len:
            # If video is shorter than desired clip, sample what we can
            indices = np.linspace(0, len(vr) - 1, frames_per_clip).astype(np.int32)
        else:
            # Sample from the middle of the video
            start = (len(vr) - clip_len) // 2
            indices = np.arange(start, start + clip_len, frame_step, dtype=np.int32)
        
        frames = vr.get_batch(indices).asnumpy()  # [T, H, W, 3] in uint8
        return frames
    except Exception as e:
        logger.warning(f"Failed to load video {video_path}: {e}")
        return None


def generate_gif(frames: np.ndarray, output_path: str, fps: float = 10.0):
    """Generate animated GIF from frames."""
    # Ensure frames are uint8 in range [0, 255]
    if frames.dtype != np.uint8:
        frames = (frames * 255).astype(np.uint8)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert to PIL images and save as GIF
    pil_images = [Image.fromarray(frame) for frame in frames]
    duration = int(1000 / fps)  # Duration per frame in milliseconds
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=0,
    )
    logger.info(f"Saved GIF to {output_path}")


def gif_to_base64(gif_path: str) -> str:
    """Convert GIF to base64 string for embedding in HTML."""
    with open(gif_path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def generate_html_report(
    videos_data: List[Dict],
    verb_labels: Dict[int, str],
    noun_labels: Dict[int, str],
    output_path: str = "evals/output/comparison_report.html"
):
    """Generate interactive HTML comparison report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Build HTML
    html_parts = []
    
    # CSS styling
    html_parts.append("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>V-JEPA Model Comparison Report</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .video-container {
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                page-break-inside: avoid;
            }
            .video-header {
                display: grid;
                grid-template-columns: 300px 1fr;
                gap: 20px;
                margin-bottom: 20px;
            }
            .video-gif {
                border: 2px solid #ddd;
                border-radius: 4px;
                max-width: 300px;
            }
            .ground-truth {
                padding: 15px;
                background-color: #f9f9f9;
                border-left: 4px solid #007bff;
            }
            .gt-label {
                margin: 8px 0;
                font-size: 14px;
            }
            .gt-title {
                font-weight: bold;
                font-size: 16px;
                color: #007bff;
                margin-bottom: 10px;
            }
            .predictions-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 15px;
                margin-top: 20px;
            }
            .model-predictions {
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                padding: 12px;
                background-color: #fafafa;
            }
            .model-name {
                font-weight: bold;
                font-size: 13px;
                color: #333;
                margin-bottom: 10px;
                padding-bottom: 8px;
                border-bottom: 2px solid #ddd;
            }
            .task-predictions {
                margin-bottom: 10px;
            }
            .task-title {
                font-weight: 600;
                font-size: 12px;
                color: #555;
                text-transform: uppercase;
                margin-bottom: 6px;
            }
            .prediction-item {
                display: flex;
                align-items: center;
                margin: 4px 0;
                font-size: 12px;
            }
            .pred-rank {
                min-width: 20px;
                color: #999;
                margin-right: 8px;
            }
            .pred-name {
                flex: 1;
                margin-right: 8px;
            }
            .pred-confidence {
                min-width: 45px;
                text-align: right;
                font-weight: 600;
            }
            .correct {
                background-color: #d4edda;
                border-left: 3px solid #28a745;
            }
            .incorrect {
                background-color: #f8d7da;
                border-left: 3px solid #dc3545;
            }
            .neutral {
                background-color: #e7e7e7;
                border-left: 3px solid #999;
            }
            .video-id {
                font-size: 12px;
                color: #999;
                margin-top: 10px;
            }
            @media print {
                .video-container {
                    page-break-inside: avoid;
                }
            }
        </style>
    </head>
    <body>
        <h1>V-JEPA Multi-Model Comparison Report</h1>
        <p style="text-align: center; color: #666;">
            Comparing 8 models on 10 randomly sampled EK100 validation videos
        </p>
    """)
    
    # Video sections
    for video_idx, video_data in enumerate(videos_data, 1):
        gif_b64 = gif_to_base64(video_data["gif_path"])
        
        verb_name = verb_labels.get(video_data["verb_id"], f"Verb {video_data['verb_id']}")
        noun_name = noun_labels.get(video_data["noun_id"], f"Noun {video_data['noun_id']}")
        
        html_parts.append(f"""
        <div class="video-container">
            <div class="video-header">
                <div>
                    <img src="data:image/gif;base64,{gif_b64}" class="video-gif" alt="Video {video_idx}">
                    <div class="video-id">Sample {video_idx}</div>
                </div>
                <div class="ground-truth">
                    <div class="gt-title">Ground Truth Labels</div>
                    <div class="gt-label"><strong>Verb:</strong> {verb_name} (ID: {video_data['verb_id']})</div>
                    <div class="gt-label"><strong>Noun:</strong> {noun_name} (ID: {video_data['noun_id']})</div>
                </div>
            </div>
            
            <div class="predictions-grid">
        """)
        
        # Model predictions
        for model_name, predictions in video_data["predictions"].items():
            html_parts.append(f"""
                <div class="model-predictions">
                    <div class="model-name">{model_name}</div>
            """)
            
            # Verb predictions
            html_parts.append("""
                    <div class="task-predictions">
                        <div class="task-title">Verb Top-5</div>
            """)
            
            for rank, (verb_id, confidence) in enumerate(predictions["verbs"], 1):
                verb_id = str(verb_id)
                verb_name_pred = verb_labels.get(verb_id, f"Verb {verb_id}")
                is_correct = verb_id == video_data["verb_id"]
                css_class = "correct" if is_correct else ("neutral" if rank == 1 else "neutral")
                
                html_parts.append(f"""
                        <div class="prediction-item {css_class}">
                            <div class="pred-rank">#{rank}</div>
                            <div class="pred-name">{verb_name_pred}</div>
                            <div class="pred-confidence">{confidence*100:.1f}%</div>
                        </div>
                """)
            
            html_parts.append("""
                    </div>
            """)
            
            # Noun predictions
            html_parts.append("""
                    <div class="task-predictions">
                        <div class="task-title">Noun Top-5</div>
            """)
            
            for rank, (noun_id, confidence) in enumerate(predictions["nouns"], 1):
                noun_id = str(noun_id)
                noun_name_pred = noun_labels.get(noun_id, f"Noun {noun_id}")
                is_correct = noun_id == video_data["noun_id"]
                css_class = "correct" if is_correct else ("neutral" if rank == 1 else "neutral")
                
                html_parts.append(f"""
                        <div class="prediction-item {css_class}">
                            <div class="pred-rank">#{rank}</div>
                            <div class="pred-name">{noun_name_pred}</div>
                            <div class="pred-confidence">{confidence*100:.1f}%</div>
                        </div>
                """)
            
            html_parts.append("""
                    </div>
                </div>
            """)
        
        html_parts.append("""
            </div>
        </div>
        """)
    
    # Close HTML
    html_parts.append("""
    </body>
    </html>
    """)
    
    html_content = "".join(html_parts)
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    logger.info(f"HTML report saved to {output_path}")


# ============================================================================
# Main Evaluation Loop
# ============================================================================

def main():
    """Main evaluation pipeline."""
    # Create output directory
    output_dir = Path("preliminary_experiments/evals/vitl/vjepa_ek100_ar/qualitative_eval")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configure logging before emitting output
    configure_logging(str(output_dir / "comparison_qualitative_report.log"))

    # Setup device
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Download class labels
    verb_labels = download_class_labels(
        VERB_CLASSES_URL,
        cache_file=str(output_dir / "verb_labels.json")
    )
    noun_labels = download_class_labels(
        NOUN_CLASSES_URL,
        cache_file=str(output_dir / "noun_labels.json")
    )
    logger.info(f"Loaded {len(verb_labels)} verb labels and {len(noun_labels)} noun labels")
    
    # Load validation data
    val_csv = "data/EK100_action_recognition_validation.csv"
    df_validation = load_validation_csv(val_csv)
    logger.info(f"Loaded {len(df_validation)} validation samples")
    
    # Sample videos
    df_samples = sample_random_videos(df_validation, n_samples=10)
    logger.info(f"Sampled {len(df_samples)} videos for evaluation")
    data_path = str(output_dir / "samples.csv")
    save_validation_csv(df_samples, data_path)
    
    
    # Prepare model configs and load models
    logger.info("=" * 80)
    logger.info("Loading models...")
    logger.info("=" * 80)
    
    models_data = []
    predictions_all_models = {}
    for model_cfg in MODELS_CONFIG:
        try:
            config = load_config(model_cfg["config"])
            encoder = init_module(
                config['model_kwargs']['module_name'],
                device,
                config['experiment']['data']['frames_per_clip'],
                config['experiment']['data']['resolution'],
                config['model_kwargs']['checkpoint'],
                config['model_kwargs']['pretrain_kwargs'],
                config['model_kwargs']['wrapper_kwargs'],
            )
            
            classifiers = load_model_checkpoint(config, encoder.embed_dim, device=device)
            model_name = model_cfg["name"]
            models_data.append({
                "name": model_name,
                "config": config,
            })
            logger.info(f"✓ Loaded: {model_cfg['name']}")
        except Exception as e:
            logger.error(f"✗ Failed to load {model_cfg['name']}: {e}")
            continue
        
        # Load data
        data_loader = load_data_for_model(config, data_path, batch_size=10)


        # Forward pass through encoder
        num_classes = config['experiment']['data']['num_classes']
        predictions_all_models[model_name] = {
                        "verbs": [],
                        "nouns": [],
                    }
        
        for itr, data in tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc="Processing"
        ):
            with torch.no_grad():
                try:
                    # Load data and put on GPU
                    clips = [
                        [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                        for di in data[0]  # iterate over temporal index of clip
                    ]
                    clip_indices = data[2]
                    if clip_indices is not None:
                        clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
                    labels = data[1] # Loaded in CPU
                    attn_mask = [
                        [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                        for di in data[3]  # iterate over temporal index of clip
                    ]
                    outputs, attn_mask = encoder(clips, clip_indices, attn_mask)
                    outputs = [[c(o, attn_mask=view_attn_mask) for o, view_attn_mask in zip(outputs, attn_mask)] for c in classifiers]

                    # Get top labels
                    outputs_ = [[[] for _ in range(len(outputs))] for _ in num_classes]
                    for c, (classifier_output) in enumerate(outputs):
                        for v, view_output in enumerate(classifier_output):
                            for t, task_output in enumerate(view_output):
                                outputs_[t][c].append(F.softmax(task_output, dim=1))
                        for t in range(len(num_classes)):
                            outputs_[t][c] = sum(outputs_[t][c]) / len(outputs_[t][c])

                    best_classifiers = []
                    for i, (task_output, task) in enumerate(zip(outputs_, ['verbs', 'nouns'])):
                        classifiers_num = len(task_output)
                        classifiers_output = torch.stack(task_output, dim=0).cpu()
                        # classifiers_output_perf = torch.gather(classifiers_output, 2, labels[i].unsqueeze(0).unsqueeze(2).repeat(classifiers_num,1,1))
                        classifiers_output_perf = (torch.argmax(classifiers_output, dim=2) == labels[i]).sum(dim=1) / classifiers_output.shape[1]
                        # classifiers_output_perf = classifiers_output_perf.squeeze().mean(dim=1)
                        best_classifier = np.nanargmax(classifiers_output_perf)
                        best_classifiers.append(classifiers_output[best_classifier])

                    for task_output, task in zip(best_classifiers, ['verbs', 'nouns']):
                        values, indices = torch.sort(task_output, dim=1, descending=True)
                        
                        for i in range(len(indices)): # batch size
                            predictions_all_models[model_name][task].extend([zip(indices[i,:5].cpu().tolist(), values[i,:5].cpu().tolist())])
                        
                    
                except Exception as e:
                    logger.error(f"    ✗ {model_name}: {e}")
                    
                    
                    
        logger.info(f"    ✓ {model_name}")
                
    
    if not models_data:
        logger.error("No models loaded successfully!")
        return
    
    videos_output = []
    
    for sample_idx, sample_row in df_samples.iterrows():
        logger.info(f"\nProcessing video {sample_idx + 1}/{len(df_samples)}")
        
        video_path = sample_row["video_path"]
        verb_id = str(sample_row["verb_id"])
        noun_id = str(sample_row["noun_id"])
        
        # Load frames
        frames = load_video_frames(video_path, frames_per_clip=32, frame_step=1)
        if frames is None:
            logger.warning(f"Skipping video {video_path}")
            continue
        
        logger.info(f"  Loaded video: {video_path}, shape: {frames.shape}")
        
        # Generate GIF
        gif_path = output_dir / f"video_{sample_idx:02d}.gif"
        generate_gif(frames, str(gif_path), fps=10.0)
        
        predictions_all_models_by_video = {}
        for key,val in predictions_all_models.items():
            predictions_all_models_by_video[key] = {}
            predictions_all_models_by_video[key]['verbs'] = val['verbs'][sample_idx]
            predictions_all_models_by_video[key]['nouns'] = val['nouns'][sample_idx]
        
        videos_output.append({
            "gif_path": str(gif_path),
            "verb_id": verb_id,
            "noun_id": noun_id,
            "predictions": predictions_all_models_by_video,
        })
    
    # Generate HTML report
    logger.info("=" * 80)
    logger.info("Generating HTML report...")
    logger.info("=" * 80)
    
    html_output = output_dir / "comparison_report.html"
    generate_html_report(videos_output, verb_labels, noun_labels, output_path=str(html_output))
    
    logger.info("=" * 80)
    logger.info(f"✓ Report generated successfully!")
    logger.info(f"  - GIFs saved to: {output_dir}/video_*.gif")
    logger.info(f"  - HTML report: {html_output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
