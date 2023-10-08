import subprocess
import json
from openbabel import openbabel
import os
import argparse
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import seaborn as sns


class binana_command():
    def __init__(self, rec_filename, lig_filename, temp_dir, unique_id, if_all=False):
        self.rec_filename = rec_filename
        self.lig_filename = lig_filename
        self.temp_dir = temp_dir
        self.unique_id = unique_id
        self.if_all = if_all
        # mk temp dir
        self.temp_dir = os.path.join(self.temp_dir, self.unique_id)
        self.temp_json = os.path.join(self.temp_dir, 'temp.json')
        os.makedirs(self.temp_dir, exist_ok=True)

        self.temp_rec_file = ''
        self.temp_lig_file = ''
        self.command = ''
        self.process_to_pdbqt()
        self.bulid_command()

    def bulid_command(self):
        self.command = f'python -m scripts.binana_script.run_binana -receptor ' + self.temp_rec_file + ' -ligand ' + self.temp_lig_file
        if self.if_all:
            self.command += ' -output_dir ' + self.temp_dir
        else:
            self.command += ' -output_json ' + self.temp_json

    def process_to_pdbqt(self):
        # process_receptor
        mol = openbabel.OBMol()
        if self.rec_filename.endswith('.pdbqt'):
            self.temp_rec_file = self.rec_filename
        elif self.rec_filename.endswith('.pdb'):
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("pdb", "pdbqt")
            obConversion.ReadFile(mol, self.rec_filename)

            temp_rec_file = os.path.join(self.temp_dir, 'rec.pdbqt')
            self.temp_rec_file = temp_rec_file
            obConversion.WriteFile(mol, temp_rec_file)

        # process_ligand
        mol = openbabel.OBMol()
        if self.lig_filename.endswith('.pdbqt'):
            self.temp_lig_file = self.lig_filename
        elif self.lig_filename.endswith('.sdf'):
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("sdf", "pdbqt")
            obConversion.ReadFile(mol, self.lig_filename)

            temp_lig_file = os.path.join(self.temp_dir, 'lig.pdbqt')
            self.temp_lig_file = temp_lig_file
            obConversion.WriteFile(mol, temp_lig_file)

    def run(self):
        subprocess.check_output(self.command, shell=True)

    def get_josn(self):
        with open(self.temp_json, 'r') as f:
            json_block = f.read()
            json_data = json.loads(json_block)
            result_data = dict(json_data)
            return result_data


def process_interactions(data):
    interactions = ['cationPiInteractions', 'halogenBonds', 'hydrogenBonds', 'piPiStackingInteractions', 'saltBridges']
    interactions = interactions[:-1]
    my_dict = {item: [] for item in interactions}
    for key in interactions:
        res_ids = []
        for inter in data[key]:
            # inter keys: ligandAtoms, metrics, receptorAtoms
            res_ids_ = []
            for res_atom in inter['receptorAtoms']:
                res_ids_.append(res_atom['resID'])
            if len(np.unique(res_ids_)) == 1:
                res_ids.append(np.unique(res_ids_).item())
        my_dict[key] += res_ids
    return my_dict


def run_binana_command(arg):
    rec, lig, temp_dir, unique_id = arg
    try:
        bn = binana_command(rec_filename=rec, lig_filename=lig, temp_dir=temp_dir, unique_id=unique_id)
        bn.run()
        data = bn.get_josn()
        return data
    except Exception as e:
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec_path', type=str, default='data/GPR84/8J18/GPR84_agonist_8J18_pocket.pdb')
    parser.add_argument('--lig_path', type=str, default='outputs_pdb/00/sdf')
    parser.add_argument('--temp_path', type=str, default='temp')
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    lig_list = []
    if not os.path.isfile(args.lig_path):
        for file in os.listdir(args.lig_path):
            file_path = os.path.join(args.lig_path, file)
            if os.path.isfile(file_path):
                lig_list.append(file_path)
    else:
        lig_list.append(args.lig_path)

    rec_list = []
    if not os.path.isfile(args.rec_path):
        for file in os.listdir(args.rec_path):
            file_path = os.path.join(args.rec_path, file)
            if os.path.isfile(file_path):
                rec_list.append(file_path)
    else:
        rec_list.append(args.rec_path)
    rec_lig_pairs = []
    for i, rec in enumerate(rec_list):
        for j, lig in enumerate(lig_list):
            rec_lig_pairs.append((rec, lig, args.temp_path, str(i) + str(j)))

    pool = Pool(processes=args.num_workers)
    results = pool.imap_unordered(run_binana_command, rec_lig_pairs)

    unique_dict = {'cationPiInteractions': [], 'halogenBonds': [], 'hydrogenBonds': [], 'piPiStackingInteractions': [], 'saltBridges': []}
    for result in results:
        if result:
            result = process_interactions(result)
            for k, v in result.items():
                unique_dict[k] += v
    print(unique_dict)

    res_ids = set()
    for k, v in unique_dict.items():
        res_ids = res_ids.union(v)
    res_ids = sorted(res_ids)
    # 创建一个空矩阵来存储频数
    interaction_matrix = np.zeros((len(res_ids), len(unique_dict.keys())))

    # 填充矩阵的相应位置
    for i, interaction_type in enumerate(unique_dict.keys()):
        for residue in unique_dict[interaction_type]:
            interaction_matrix[res_ids.index(residue), i] += 1

    # 设置图形大小
    plt.figure(figsize=(10, 8), dpi=300)

    # 使用seaborn库的热力图函数绘制热力图
    sns.heatmap(interaction_matrix, cmap='Blues', annot=True, cbar=True, fmt='.0f')

    # 设置坐标轴标签
    plt.xlabel('Interaction Type', fontdict={'fontsize': 15})
    plt.ylabel('Residue Number', fontdict={'fontsize': 15})

    # 设置相互作用类型标签
    interaction_types = list(unique_dict.keys())
    plt.xticks(np.arange(len(interaction_types)) + 0.5, interaction_types)
    plt.yticks(np.arange(len(res_ids)) + 0.5, res_ids)
    plt.tight_layout()
    # 显示图形
    plt.show()
