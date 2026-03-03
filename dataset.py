import os
import torch
from torch.utils.data import Dataset
import cv2
import random
import numpy as np
from sklearn.model_selection import KFold, train_test_split

class DataSplitter:
    """
    Handles the 3-way data splitting protocol:
    1. Hold-out Test Set (Completely Isolated)
    2. Development Set -> K-Fold (Train / Val)
    """
    @staticmethod
    def get_case_ids(data_root):
        """Scan and retrieve all unique Case IDs"""
        qp_dir = os.path.join(data_root, "QP")
        if not os.path.exists(qp_dir):
            raise FileNotFoundError(f"QP directory not found: {qp_dir}")
        
        case_ids = set()
        for fname in os.listdir(qp_dir):
            if fname.lower().endswith((".jpg", ".png", ".jpeg", ".tif")):
                case_id = fname.split("_QP")[0] 
                case_ids.add(case_id)
        return sorted(list(case_ids))

    @staticmethod
    def get_split_ids(data_root, fold_idx=0, total_folds=5, holdout_ratio=0.20, seed=42):
        """
        Core splitting logic.
        Returns:
            train_ids: Used for training
            val_ids:   Used for checkpoint selection
            test_ids:  Hold-out test set for final reporting
        """
        all_ids = np.array(DataSplitter.get_case_ids(data_root))
        
        # 1. Isolate the Hold-out Test Set
        # random_state must be fixed to ensure the test set remains strictly identical across runs.
        dev_ids, test_ids = train_test_split(
            all_ids, test_size=holdout_ratio, random_state=999
        )
        
        # 2. Perform K-Fold on the Development Set
        kf = KFold(n_splits=total_folds, shuffle=True, random_state=seed)
        splits = list(kf.split(dev_ids))
        
        if fold_idx >= total_folds:
            raise ValueError(f"Fold idx {fold_idx} out of range")
            
        train_idx, val_idx = splits[fold_idx]
        train_ids = dev_ids[train_idx]
        val_ids = dev_ids[val_idx]
        
        return train_ids.tolist(), val_ids.tolist(), test_ids.tolist()

class PathologyDataset(Dataset):
    def __init__(self, data_root, case_ids, mode="train", transform=None, similar_neg_ratio=0.8, seed=31):
        self.data_root = data_root
        self.case_ids = set(case_ids)  
        self.transform = transform
        self.similar_neg_ratio = similar_neg_ratio
        self.img_size = 224
        
        random.seed(seed)
        self.qp_dir = os.path.join(data_root, "QP")
        self.lk_dir = os.path.join(data_root, "LK")
        
        self.pairs = []
        self.all_lk_candidates = [] 
        
        id_map = self._build_filtered_map()
        self._generate_pairs(id_map)
        
        if len(self.pairs) == 0:
            print(f"Warning: Dataset for mode {mode} is empty! Check case_ids.")

    def _build_filtered_map(self):
        id_map = {}
        # Scan QP
        for fname in os.listdir(self.qp_dir):
            if not fname.lower().endswith((".jpg", ".png")): continue
            case_id = fname.split("_QP")[0]
            if case_id not in self.case_ids: continue 
            
            path = os.path.join(self.qp_dir, fname)
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                id_map[case_id] = {"qp": {"path": path, "size": (w, h)}, "lk": []}

        # Scan LK
        for fname in os.listdir(self.lk_dir):
            if not fname.lower().endswith((".jpg", ".png")): continue
            case_id = fname.split("_LK")[0]
            if case_id in id_map: 
                path = os.path.join(self.lk_dir, fname)
                img = cv2.imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    info = {"path": path, "size": (w, h), "case_id": case_id}
                    id_map[case_id]["lk"].append(info)
                    self.all_lk_candidates.append(info)
        return id_map

    def _generate_pairs(self, id_map):
        valid_ids = [k for k, v in id_map.items() if v["qp"] and len(v["lk"]) > 0]
        for case_id in valid_ids:
            qp_info = id_map[case_id]["qp"]
            real_lks = id_map[case_id]["lk"]
            
            for lk_item in real_lks:
                # Positive Sample
                self.pairs.append((qp_info["path"], lk_item["path"], 1.0))
                # Negative Sample
                neg_lk = None
                if random.random() < self.similar_neg_ratio:
                    neg_lk = self._find_similar_negative(qp_info["size"], case_id)
                else:
                    neg_lk = self._find_random_negative(case_id)
                
                if neg_lk:
                    self.pairs.append((qp_info["path"], neg_lk["path"], 0.0))
                else:
                    self.pairs.pop() # Maintain balance if generation fails

    def _find_similar_negative(self, target_size, exclude_id):
        cands = [x for x in self.all_lk_candidates if x["case_id"] != exclude_id]
        if not cands: return None
        # Simple size-based similarity proxy sampling
        subset = random.sample(cands, min(len(cands), 50))
        subset.sort(key=lambda x: abs(x["size"][0]-target_size[0]) + abs(x["size"][1]-target_size[1]))
        return random.choice(subset[:5])

    def _find_random_negative(self, exclude_id):
        cands = [x for x in self.all_lk_candidates if x["case_id"] != exclude_id]
        return random.choice(cands) if cands else None

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        qp_path, lk_path, label = self.pairs[idx]
        qp_img = cv2.imread(qp_path)
        lk_img = cv2.imread(lk_path)
        
        # Exception handling for unreadable files
        if qp_img is None: qp_img = np.zeros((224, 224, 3), dtype=np.uint8)
        if lk_img is None: lk_img = np.zeros((224, 224, 3), dtype=np.uint8)
            
        qp_img = cv2.cvtColor(qp_img, cv2.COLOR_BGR2RGB)
        lk_img = cv2.cvtColor(lk_img, cv2.COLOR_BGR2RGB)
        
        qp_img, lk_img = self._smart_resize_and_pad(qp_img, lk_img, self.img_size)
        
        if self.transform:
            try:
                res = self.transform(image=qp_img, image2=lk_img)
                qp_img, lk_img = res["image"], res["image2"]
            except Exception:
                pass 
                
        return qp_img, lk_img, torch.tensor(label, dtype=torch.float32)

    def _smart_resize_and_pad(self, img1, img2, target_size):
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        max_side = max(h1, w1, h2, w2)
        
        if max_side > target_size:
            scale = target_size / max_side
            img1 = cv2.resize(img1, (int(w1*scale), int(h1*scale)), interpolation=cv2.INTER_AREA)
            img2 = cv2.resize(img2, (int(w2*scale), int(h2*scale)), interpolation=cv2.INTER_AREA)
            
        return self._pad_image(img1, target_size), self._pad_image(img2, target_size)

    def _pad_image(self, img, target_size):
        h, w = img.shape[:2]
        pad_h, pad_w = max(target_size - h, 0), max(target_size - w, 0)
        return cv2.copyMakeBorder(img, pad_h//2, pad_h-pad_h//2, pad_w//2, pad_w-pad_w//2, cv2.BORDER_CONSTANT, value=0)
