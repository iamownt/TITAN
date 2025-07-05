import os
import h5py
import torch
import pickle
from pathlib import Path
from transformers import AutoModel
import numpy as np
from tqdm import tqdm
from typing import Union, List
import torch.multiprocessing as mp

from collections import defaultdict
import torch
import torch.multiprocessing as mp
from transformers import AutoModel
import h5py
import os
from pathlib import Path
from tqdm import tqdm
from typing import List
import argparse
import pandas as pd


def get_tile_count(titan_tile_path: str = "/home/user/sngp/project/titan_destination_20X/h5file",
                   feature_name: str = "tile_embeds"):
    # file name to each OnTreeCode So that we can get each tile count for each particular slide
    train_split = pd.read_csv('/home/user/wangtao/prov-gigapath/TITAN/datasets/tcga-ot_train.csv')
    val_split = pd.read_csv('/home/user/wangtao/prov-gigapath/TITAN/datasets/tcga-ot_val.csv')
    test_split = pd.read_csv('/home/user/wangtao/prov-gigapath/TITAN/datasets/tcga-ot_test.csv')
    df_concat = pd.concat([train_split, val_split, test_split], axis=0)

    df_concat['slide_id'] = df_concat['slide_id'].str.replace(
        'TCGA-F5-6861-01Z-00-DX1.011b771b-f52e-412e-9352-1578349beaf1',
        'TCGA-F5-6861-01Z-00-DX1.011B771B-F52E-412E-9352-1578349BEAF1',
        regex=False
    )
    print(f"train split {train_split.shape}, val split {val_split.shape}, test split {test_split.shape}")

    filename_to_OncoTreeCode_map = df_concat.set_index("slide_id")["OncoTreeCode"].to_dict()


    titan_tile_path = Path(titan_tile_path)
    file_path = os.listdir(titan_tile_path)
    tile_count = []
    OncoTreeCode_map = defaultdict(list)
    for file in tqdm(file_path, desc="Processing Slide embedding"):
        demo_h5_path = titan_tile_path / file
        with h5py.File(demo_h5_path, 'r') as f:
            tile_count.append(f[feature_name].shape[0])
        OncoTreeCode_map[filename_to_OncoTreeCode_map[file[:-3]]].append(tile_count[-1])
    print("Average Tile Count Per Slide", np.mean(tile_count))
    return tile_count, OncoTreeCode_map


def inference_slide_embedding_mp(titan_tile_path: str = "/home/user/sngp/project/titan_destination_20X/h5file",
                                 titan_slide_path: str = "/home/user/sngp/project/titan_destination_20X/h5file_slide",
                                 feature_name: str = "tile_embeds", no_auto_skip: bool = False, gpu_id: int = 0,
                                 total_gpus: int = 2):
    device = torch.device(f"cuda:{gpu_id}")
    model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    model = model.to(device)
    model.eval()


    titan_tile_path = Path(titan_tile_path)
    titan_slide_path = Path(titan_slide_path)
    titan_slide_path.mkdir(parents=True, exist_ok=True)
    patch_size_level0 = torch.tensor([512]).to(device)
    file_path = os.listdir(titan_tile_path)
    file_path.sort() # Ensure consistent file order across processes
    assigned_files = [f for i, f in enumerate(file_path) if i % total_gpus == gpu_id]

    for file in tqdm(assigned_files, desc="Processing Slide embedding"):
        demo_h5_path = titan_tile_path / file
        slide_h5_path = titan_slide_path / file

        if os.path.exists(slide_h5_path) and not no_auto_skip:
            continue
        print(demo_h5_path)
        with h5py.File(demo_h5_path, 'r') as f:
            features = torch.from_numpy(f[feature_name][:]).unsqueeze(dim=0)
            coords = torch.from_numpy(f['coords'][:]).unsqueeze(dim=0).long()
        print("features", features.shape)
        print("coords", coords.shape)

        # extract slide embedding
        try:
            with torch.autocast(device_type='cuda', dtype=torch.float16), torch.inference_mode():
                features = features.to(device)
                coords = coords.to(device)
                slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_level0)
                print("slide_embedding", slide_embedding.shape)
        except torch.OutOfMemoryError: # utilize cpu
            cpu_device = torch.device("cpu")
            with torch.autocast(device_type='cpu', dtype=torch.float16), torch.inference_mode():
                features = features.to(cpu_device)
                coords = coords.to(cpu_device)
                model.to(cpu_device)
                slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_level0.cpu())
                print("[CPU] slide_embedding", slide_embedding.shape)
            model.to(device)

        with h5py.File(slide_h5_path, 'w') as f:
            f.create_dataset('slide_embedding', data=slide_embedding.detach().float().cpu().numpy())
        del features, coords, slide_embedding

def merge_slide_embedding(pkl_path: Union[str, None] = "/home/user/sngp/project/destination/pickle_data.pkl",
                          titan_slide_path: str = "/home/user/sngp/project/titan_destination_20X/h5file_slide",
                          dataset_name: str = "tcga", subtyping: str = "nsclc"):
    if pkl_path is not None:
        with open(pkl_path, "rb") as f:
            pkl_dict = pickle.load(f)
        invert_pkl_dict = {v: k for k, v in pkl_dict.items()}
    titan_slide_path = Path(titan_slide_path)
    slide_embedding, slide_id = [], []

    for file in os.listdir(titan_slide_path):
        slide_h5_path = titan_slide_path / file
        print("slide_h5_path", slide_h5_path)
        with h5py.File(slide_h5_path, 'r') as f:
            embed = f['slide_embedding'][:]
            slide_embedding.append(embed)
            if pkl_path is not None:
                slide_name = invert_pkl_dict[int(file[:-3])]
            else:
                slide_name = file[:-3]
            slide_id.append(slide_name)
    slide_embedding = np.concatenate(slide_embedding, axis=0)
    # with open(titan_slide_path / f"{dataset_name}_{subtyping}_titan_slide_embedding.pkl", "wb") as f:
    #     pickle.dump({"embeddings": slide_embedding, "filenames": slide_id}, f)


def inference_slide_embedding_eat(trials: int = 1, folds: int = 5,
                                  keep_ratio: float = 0.4,
                                  titan_tile_path: str = "/home/user/sngp/project/titan_destination_20X/h5file",
                                  titan_slide_path: str = "/home/user/sngp/project/titan_destination_20X/h5file_slide_eat",
                                  titan_amb_path: str = "/home/user/sngp/UniConch/models/ambpkl/newambk/titan_itest_ambiguity_dict_autogluon_0.2_tuning0.pkl"):
    with open(titan_amb_path, "rb") as f:
        titan_amb = pickle.load(f)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    model = model.to(device)
    model.eval()


    titan_tile_path = Path(titan_tile_path)
    titan_slide_path = Path(titan_slide_path)
    titan_slide_path.mkdir(parents=True, exist_ok=True)
    patch_size_level0 = torch.tensor([512]).to(device)
    file_path = os.listdir(titan_tile_path)
    for trial in range(trials):
        for fold in range(folds):
            for file in tqdm(file_path, desc="Processing Slide embedding"):
                demo_h5_path = titan_tile_path / file
                print(demo_h5_path)
                with h5py.File(demo_h5_path, 'r') as f:
                    features = torch.from_numpy(f['tile_embeds'][:]).unsqueeze(dim=0)
                    coords = torch.from_numpy(f['coords'][:]).unsqueeze(dim=0).long()
                print("features", features.shape)
                print("coords", coords.shape)
                amb_array = titan_amb[f"t{trial}f{fold}"][file[:-3]]
                in_slide_threshold = np.quantile(amb_array, keep_ratio)
                mask_bool = amb_array <= in_slide_threshold
                # extract slide embedding
                with torch.autocast(device_type='cuda', dtype=torch.float16), torch.inference_mode():
                    features = features.to(device)
                    coords = coords.to(device)
                    features = features[:, mask_bool]
                    coords = coords[:, mask_bool]
                    print("reduce from {} to {}".format(len(mask_bool), len(mask_bool[mask_bool])))
                    slide_embedding = model.encode_slide_from_patch_features(features, coords, patch_size_level0)
                    print("slide_embedding", slide_embedding.shape)

                slide_h5_path = titan_slide_path / f"t{trial}f{fold}"
                slide_h5_path.mkdir(parents=True, exist_ok=True)
                with h5py.File(slide_h5_path / file, 'w') as f:
                    f.create_dataset('slide_embedding', data=slide_embedding.detach().float().cpu().numpy())

def merge_slide_embedding_eat(trials: int = 1, folds: int = 5,
                              pkl_path: str = "/home/user/sngp/project/destination/pickle_data.pkl",
                              titan_slide_path: str = "/home/user/sngp/project/titan_destination_20X/h5file_slide_eat",
                              dataset_name: str = "tcga", subtyping: str = "nsclc"):
    with open(pkl_path, "rb") as f:
        pkl_dict = pickle.load(f)
    invert_pkl_dict = {v: k for k, v in pkl_dict.items()}
    titan_slide_path = Path(titan_slide_path)
    for trial in range(trials):
        for fold in range(folds):
            slide_embedding, slide_id = [], []
            for file in os.listdir(titan_slide_path / f"t{trial}f{fold}"):
                if "slide_embedding" in file:
                    continue
                slide_h5_path = titan_slide_path / f"t{trial}f{fold}" / file
                with h5py.File(slide_h5_path, 'r') as f:
                    slide_embedding.append(f['slide_embedding'][:])
                    slide_id.append(invert_pkl_dict[int(file[:-3])])

            slide_embedding = np.concatenate(slide_embedding, axis=0)
            with open(titan_slide_path / f"t{trial}f{fold}" / f"{dataset_name}_{subtyping}_titan_slide_embedding.pkl", "wb") as f:
                pickle.dump({"embeddings": slide_embedding, "filenames": slide_id}, f)


if __name__ == "__main__":
    # clam based features, performance
    parser = argparse.ArgumentParser(description='ABMIL on downstream tasks')
    parser.add_argument('--gpu_id',                  type=int,  default=0,  help='GPU ID')
    parser.add_argument('--total_gpus',                   type=int,  default=1,  help='Number of GPU')

    args = parser.parse_args()
    titan_tile_path = "/home/user/sngp/TCGA-OT/h5_files"
    titan_slide_path = "/home/user/sngp/TCGA-OT/h5_files_slide"
    pkl_path = None
    feature_name = "features"
    tile_count,  OncoTreeCode_map = get_tile_count(titan_tile_path=titan_tile_path, feature_name=feature_name)
    # plot
    import matplotlib.pyplot as plt
    # plt.hist(tile_count, bins=100)
    # plt.xlabel("Tile Count")
    # plt.show()

    average_tile_counts = {dataset: np.mean(tile_counts) for dataset, tile_counts in OncoTreeCode_map.items()}

    XYTICK_FONTSIZE = 22
    LEGNED_FONTSIZE = 26
    XYLABEL_FONTSIZE = 26
    color_ax = '#3a3a3a'  # r, g, b = 58, 58, 58
    color_bar = '#505050'  # r, g, b = 80, 80, 80
    colors = ['#89CFF0', '#bdbdbd', '#75A9CB', 'lightblue', '#bdbdbd', '#89CFF0', '#bdbdbd', '#89CFF0', '#89CFF0']
    # Plotting the bar plot
    fig, ax = plt.subplots(1, 1, figsize=(25, 8))
    ax.bar(average_tile_counts.keys(), average_tile_counts.values())
    ax.set_xlabel('Dataset Name', fontsize=XYLABEL_FONTSIZE, color=color_ax)
    ax.set_ylabel('Average Tile Count', fontsize=XYLABEL_FONTSIZE, color=color_ax)
    ax.set_title('Average Tile Count per Dataset', fontsize=XYLABEL_FONTSIZE, color=color_ax)
    ax.tick_params(labelsize=XYTICK_FONTSIZE, rotation=45, length=10, width=0, color=color_bar)

    plt.tight_layout()
    plt.show()


    #
    # inference_slide_embedding_mp(titan_tile_path=titan_tile_path, titan_slide_path=titan_slide_path,
    #                              feature_name=feature_name, no_auto_skip=False, gpu_id=args.gpu_id,
    #                              total_gpus=args.total_gpus)
    # merge_slide_embedding(pkl_path=pkl_path, titan_slide_path=titan_slide_path,
    #                       dataset_name="tcga", subtyping="nsclc")
    # no multiprocessing
    # inference_slide_embedding(titan_tile_path=titan_tile_path, titan_slide_path=titan_slide_path,
    #                           feature_name=feature_name, no_auto_skip=True)
    # merge_slide_embedding(pkl_path=pkl_path, titan_slide_path=titan_slide_path,
    #                       dataset_name="tcga", subtyping="nsclc")


    # normal[previous] generation, previous slideflow based features
    # inference_slide_embedding(titan_tile_path="/home/user/sngp/project/titan_cptac_destination_20X/h5file",
    #                           titan_slide_path="/home/user/sngp/project/titan_cptac_destination_20X/h5file_slide")
    # merge_slide_embedding(pkl_path="/home/user/sngp/project/cptac_destination/pickle_data.pkl",
    #                         titan_slide_path="/home/user/sngp/project/titan_cptac_destination_20X/h5file_slide",
    #                         dataset_name="cptac", subtyping="nsclc")
    # eat version
    # for tune_mask_ratio in [0.4]:
    #     inference_slide_embedding_eat(trials=4, folds=5, keep_ratio=tune_mask_ratio,
    #                                   titan_tile_path="/home/user/sngp/project/titan_destination_20X/h5file",
    #                                   titan_slide_path=f"/home/user/sngp/project/titan_destination_20X/h5file_slide_eat{tune_mask_ratio}",
    #                                   titan_amb_path="/home/user/sngp/UniConch/models/ambpkl/newambk/titan_itest_ambiguity_dict_autogluon_1.0_tuning0.pkl")
    #     merge_slide_embedding_eat(trials=4, folds=5, pkl_path=f"/home/user/sngp/project/destination/pickle_data.pkl",
    #                               titan_slide_path=f"/home/user/sngp/project/titan_destination_20X/h5file_slide_eat{tune_mask_ratio}",
    #                               dataset_name="tcga", subtyping="nsclc")



    # -------------------------------------- previous tcga version --------------------------------------
    # non-eat version, deal with the ood dataset
    # for ood_dataset in ["blca", "ucs", "uvm", "acc"]:
    #     inference_slide_embedding(titan_tile_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file",
    #                               titan_slide_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file_slide")
    #     merge_slide_embedding(pkl_path=f"/home/user/sngp/UniConch/ood_pkl_folder/{ood_dataset}_pickle_data.pkl",
    #                           titan_slide_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file_slide",
    #                           dataset_name=ood_dataset)
    # eat version, deal with the ood dataset
    # tune_mask_ratio = 0.8
    # for ood_dataset in ["blca", "ucs", "uvm", "acc"]:
    #     inference_slide_embedding_eat(trials=4, folds=5, keep_ratio=tune_mask_ratio,
    #                                   titan_tile_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file",
    #                                   titan_slide_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file_slide_eat{tune_mask_ratio}",
    #                                   titan_amb_path=f"/home/user/sngp/UniConch/models/ambpkl/newambk/titan_{ood_dataset}_ambiguity_dict_autogluon_1.0_tuning0.pkl")
    #     merge_slide_embedding_eat(trials=4, folds=5, pkl_path=f"/home/user/sngp/UniConch/ood_pkl_folder/{ood_dataset}_pickle_data.pkl",
    #                               titan_slide_path=f"/home/user/sngp/UniConch/titan_{ood_dataset}_h5file_slide_eat{tune_mask_ratio}",
    #                               dataset_name=ood_dataset)