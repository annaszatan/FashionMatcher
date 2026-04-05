import os
import sys
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from torchvision import transforms
import torch
import yaml
from PIL import Image

from ml.models.embedding_models import build_embedding_model


class RetrievalModel:
	"""FAISS-backed retrieval wrapper for gallery search.
	Requires: 
	- A FAISS index file containing the gallery embeddings and their corresponding metadata arrays (paths, item_ids, product_names, source_urls).
	- A trained embedding model checkpoint and config for projecting query embeddings if using dynamic query encoding from
	"""

	def __init__(
		self,
		index_path: str,
		gallery_paths_path: Optional[str] = None,
		gallery_item_ids_path: Optional[str] = None,
		gallery_product_names_path: Optional[str] = None,
		gallery_source_urls_path: Optional[str] = None,
	):
		self.index_path = index_path
		self.gallery_paths = self.load_optional_array(gallery_paths_path)
		self.gallery_item_ids = self.load_optional_array(gallery_item_ids_path)
		self.gallery_product_names = self.load_optional_array(gallery_product_names_path)
		self.gallery_source_urls = self.load_optional_array(gallery_source_urls_path)

		if not os.path.isfile(self.index_path):
			raise FileNotFoundError(f"FAISS index not found: {self.index_path}")

		self.index = faiss.read_index(self.index_path)
		self._dino_model_cache: Dict[str, Any] = {}
		self.embedding_model_cache: Dict[str, Any] = {}

	@staticmethod
	def load_optional_array(path: Optional[str]) -> Optional[np.ndarray]:
		if path is None:
			return None
		if not os.path.isfile(path):
			raise FileNotFoundError(f"Array file not found: {path}")

		ext = os.path.splitext(path)[1].lower()
		if ext == ".npy":
			return np.load(path, allow_pickle=True)

		with open(path, "r", encoding="utf-8") as f:
			lines = [line.strip() for line in f if line.strip()]
		return np.array(lines, dtype=object)

	@staticmethod
	def load_query_embeddings(query_path: str) -> np.ndarray:
		if not os.path.isfile(query_path):
			raise FileNotFoundError(f"Query embeddings not found: {query_path}")
		queries = np.load(query_path).astype(np.float32)
		return queries

	@staticmethod
	def build_transform(image_size: int = 224):

		return transforms.Compose(
			[
				transforms.Resize((image_size, image_size)),
				transforms.ToTensor(),
				transforms.Normalize(
					mean=(0.485, 0.456, 0.406),
					std=(0.229, 0.224, 0.225),
				),
			]
		)

	def load_dino_model(self, backbone_name: str, device: str):

		allowed = {"dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"}
		if backbone_name not in allowed:
			raise ValueError(f"Unknown DINOv2 backbone: {backbone_name}")

		cache_key = f"{backbone_name}|{device}"
		if cache_key in self._dino_model_cache:
			return self._dino_model_cache[cache_key]

		# torch.hub downloads the model once and uses the local cache afterward.
		model = torch.hub.load("facebookresearch/dinov2", backbone_name, pretrained=True, verbose=False)

		model.eval()
		model.to(device)
		self._dino_model_cache[cache_key] = model
		return model

	@staticmethod
	def resolve_path(path: str, project_root: Optional[str]) -> str:
		# Ensure paths are correct (either use absolute or relative to project root)
		if os.path.isabs(path) or project_root is None:
			return path
		return os.path.normpath(os.path.join(project_root, path))

	def project_with_embedding_head(
		self,
		dino_feats: np.ndarray,
		embedding_config_path: str,
		embedding_checkpoint_path: str,
		project_root: Optional[str] = None,
		embedding_device: str = "cuda",
	) -> np.ndarray:

		if embedding_device == "cuda" and not torch.cuda.is_available():
			embedding_device = "cpu"

		# Resolve the config and checkpoint paths
		cfg_path = RetrievalModel.resolve_path(embedding_config_path, project_root)
		ckpt_path = RetrievalModel.resolve_path(embedding_checkpoint_path, project_root)
		if not os.path.isfile(cfg_path):
			raise FileNotFoundError(f"Embedding config not found: {cfg_path}")
		if not os.path.isfile(ckpt_path):
			raise FileNotFoundError(f"Embedding checkpoint not found: {ckpt_path}")

		# Create the shape key for caching based on the DINO feature shape
		if dino_feats.ndim == 3:
			shape_key = f"3d|{int(dino_feats.shape[1])}|{int(dino_feats.shape[2])}"
		else:
			shape_key = f"2d|{int(dino_feats.shape[1])}"

		cache_key = f"{cfg_path}|{ckpt_path}|{embedding_device}|{shape_key}"
		if cache_key not in self.embedding_model_cache:
			with open(cfg_path, "r", encoding="utf-8") as f:
				cfg = yaml.safe_load(f)
			model_cfg = cfg.get("model", {})

			if project_root and project_root not in sys.path:
				sys.path.insert(0, project_root)

			device = torch.device(embedding_device)
			if dino_feats.ndim == 3:
				seq_len_i = int(dino_feats.shape[1])
				token_dim_i = int(dino_feats.shape[2])

				# Create the embedding model based on provided config and checkpoint
				emb_model = build_embedding_model(
					model_type=model_cfg.get("type", "transformer"),
					cfg=model_cfg,
					seq_len=seq_len_i,
					token_dim=token_dim_i,
				).to(device)
			else:
				input_dim = int(dino_feats.shape[1])
				emb_model = build_embedding_model(
					model_type=model_cfg.get("type", "transformer"),
					cfg=model_cfg,
					input_dim=input_dim,
				).to(device)

			# Load the checkpoint weights to the embedding model and freeze weights (eval)
			ckpt = torch.load(ckpt_path, map_location=device)
			emb_model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=True)
			emb_model.eval()
			self.embedding_model_cache[cache_key] = emb_model

		emb_model = self.embedding_model_cache[cache_key]
		device = next(emb_model.parameters()).device

		# Forward the DINO features through the embedding model to get the projected query embeddings
		with torch.no_grad():
			x = torch.from_numpy(dino_feats.astype(np.float32)).to(device)
			y = emb_model(x).detach().cpu().numpy().astype(np.float32)

		return y

	def encode_image_paths(
		self,
		image_paths: List[str],
		backbone_name: str = "dinov2_vitb14",
		device: str = "cuda",
		token_mode: str = "cls",
		normalize: bool = True,
		image_size: int = 224,
	) -> np.ndarray:
		"""This function encodes a list of image paths in DINO features"""
	
		if device == "cuda" and not torch.cuda.is_available():
			device = "cpu"

		model = self.load_dino_model(backbone_name, device)
		transform = RetrievalModel.build_transform(image_size=image_size)

		feats_list = []
		with torch.no_grad():
			for p in image_paths:
				if not os.path.isfile(p):
					raise FileNotFoundError(f"Query image not found: {p}")

				img = Image.open(p).convert("RGB")
				x = transform(img).unsqueeze(0).to(device, non_blocking=True)

				if token_mode == "cls":
					f = model(x).detach().float().cpu().numpy()
				elif token_mode == "patch":
					out = model.forward_features(x)
					f = out["x_norm_patchtokens"].detach().float().cpu().numpy()
				elif token_mode == "cls_patch":
					out = model.forward_features(x)
					cls_token = out["x_norm_clstoken"].unsqueeze(1)
					patch = out["x_norm_patchtokens"]
					f = torch.cat([cls_token, patch], dim=1).detach().float().cpu().numpy()
				else:
					raise ValueError('token_mode must be one of: "cls", "patch", "cls_patch"')

				feats_list.append(f)

		if len(feats_list) == 0:
			raise ValueError("No query images were provided.")

		all_feats = np.concatenate(feats_list, axis=0)

		if normalize:
			if all_feats.ndim == 2:
				norms = np.linalg.norm(all_feats, axis=1, keepdims=True) + 1e-10
				all_feats = all_feats / norms
			else:
				norms = np.linalg.norm(all_feats, axis=-1, keepdims=True) + 1e-10
				all_feats = all_feats / norms

		return all_feats.astype(np.float32)

	def search(self, query_embeddings: np.ndarray, topk: int = 5) -> List[List[Dict[str, Any]]]:
		"""
		Performs a FAISS search for the provided query embeddings and returns a list of results with metadata for each query
		"""
		if query_embeddings.ndim != 2:
			raise ValueError(f"query_embeddings must be 2D [N, D], got shape {query_embeddings.shape}")

		if query_embeddings.shape[1] != self.index.d:
			raise ValueError(
				f"Dimension mismatch: index dim={self.index.d}, query dim={query_embeddings.shape[1]}"
			)

		sims, idxs = self.index.search(query_embeddings.astype(np.float32), topk)

		all_results: List[List[Dict[str, Any]]] = []
		for qi in range(query_embeddings.shape[0]):
			query_results: List[Dict[str, Any]] = []
			for rank, (g_idx, sim) in enumerate(zip(idxs[qi], sims[qi]), start=1):
				if int(g_idx) < 0:
					continue

				rec: Dict[str, Any] = {
					"rank": rank,
					"gallery_index": int(g_idx),
					"score": float(sim),
				}

				if self.gallery_paths is not None and int(g_idx) < len(self.gallery_paths):
					img_path = str(self.gallery_paths[int(g_idx)])
					rec["path"] = img_path
					rec["image_path"] = img_path

				if self.gallery_item_ids is not None and int(g_idx) < len(self.gallery_item_ids):
					rec["item_id"] = str(self.gallery_item_ids[int(g_idx)])

				if self.gallery_product_names is not None and int(g_idx) < len(self.gallery_product_names):
					rec["product_name"] = str(self.gallery_product_names[int(g_idx)])

				if self.gallery_source_urls is not None and int(g_idx) < len(self.gallery_source_urls):
					rec["product_url"] = str(self.gallery_source_urls[int(g_idx)])

				query_results.append(rec)
			all_results.append(query_results)

		return all_results

	# Searches based on pre-encoded query embeddings loaded from an .npy file
	def search_from_npy(self, query_path: str, topk: int = 2) -> List[List[Dict[str, Any]]]:
		queries = self.load_query_embeddings(query_path)
		return self.search(queries, topk=topk)

	# Dynamic search (encode images then search)
	def search_from_images(
		self,
		image_paths: List[str],
		topk: int = 3,
		backbone_name: str = "dinov2_vitb14",
		device: str = "cuda",
		token_mode: str = "cls",
		normalize: bool = True,
		image_size: int = 224,
		project_root: Optional[str] = None,
		embedding_config_path: Optional[str] = None,
		embedding_checkpoint_path: Optional[str] = None,
		embedding_device: str = "cuda",
	) -> List[List[Dict[str, Any]]]:
		if embedding_config_path is None or embedding_checkpoint_path is None:
			raise ValueError(
				"search_from_images now requires embedding_config_path and embedding_checkpoint_path."
			)

		# Encode the provided images with DINO features
		queries = self.encode_image_paths(
			image_paths=image_paths,
			backbone_name=backbone_name,
			device=device,
			token_mode=token_mode,
			normalize=normalize,
			image_size=image_size,
		)

		# Project the DINO features through the embedding head to get final query embeddings for retrieval
		queries = self.project_with_embedding_head(
			dino_feats=queries,
			embedding_config_path=embedding_config_path,
			embedding_checkpoint_path=embedding_checkpoint_path,
			project_root=project_root,
			embedding_device=embedding_device,
		)

		if queries.ndim != 2:
			raise ValueError(
				"Dynamic query embeddings must be 2D for FAISS search. "
				"Check token_mode and embedding model config/checkpoint compatibility."
			)

		# Return search results from projected query embeddings
		return self.search(queries, topk=topk)
