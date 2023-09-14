import lmdb, pickle, re, subprocess, os, random
import torch
import argparse
from datasets.pl_data import ProteinLigandData, torchify_dict
import numpy as np
import binana, pickle
from tqdm import tqdm
from utils.transforms import *
from rdkit import Chem
from functools import partial
import pandas as pd
from multiprocessing import Pool, set_start_method

AA_NAME_SYM = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
AA_NUMBER = {i: k for i, (k, _) in enumerate(AA_NAME_SYM.items())}
LIG_ATOM = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 17: 'Cl', 15: 'P', 16: 'S', 35: 'Br', 53: 'I'}

INTERACTION = {0: 'cation_pi', 1: 'halogen', 2: 'hydrogen', 3: 'pi-pi'}
interaction_prompt = {
    'ALA_halogen': 0, 'ALA_hydrogen': 1, 'ASP_halogen': 2, 'ASP_hydrogen': 3, 'GLU_halogen': 4, 'GLU_hydrogen': 5,
    'CYS_halogen': 6, 'CYS_hydrogen': 7, 'GLY_halogen': 8, 'GLY_hydrogen': 9, 'HIS_halogen': 10, 'HIS_hydrogen': 11,
    'ILE_halogen': 12, 'ILE_hydrogen': 13, 'LYS_halogen': 14, 'LYS_hydrogen': 15, 'LEU_halogen': 16, 'LEU_hydrogen': 17,
    'MET_halogen': 18, 'MET_hydrogen': 19, 'ASN_halogen': 20, 'ASN_hydrogen': 21, 'PRO_halogen': 22, 'PRO_hydrogen': 23,
    'GLN_halogen': 24, 'GLN_hydrogen': 25, 'ARG_halogen': 26, 'ARG_hydrogen': 27, 'SER_halogen': 28, 'SER_hydrogen': 29,
    'THR_halogen': 30, 'THR_hydrogen': 31, 'VAL_halogen': 32, 'VAL_hydrogen': 33, 'TRP_halogen': 34, 'TRP_hydrogen': 35,
    'TYR_caption': 36, 'TYR_halogen': 37, 'TYR_hydrogen': 38, 'TYR_pi': 39, 'PHE_caption': 40, 'PHE_halogen': 41, 'PHE_hydrogen': 42, 'PHE_pi': 43,
    'None': 44
}


def inter_detect(data, args):
    names = data['ligand_filename'].split("/")
    orig_ligand_path = os.path.join(args.source_data_path, data['ligand_filename'])
    temp_ligand_path = os.path.join(args.temp_path, names[0] + "__" + names[1][:-4] + '.pdb')
    # obabel转换pdb文件会对配体中部分原子命名为atom,导致binana读取文件时候产生duplicate警告
    # 并且忽略该原子,这样会提取到完全错误的相互作用,改用rdkit将sdf文件转换为binana需要的pdb文件
    if os.path.exists(temp_ligand_path):
        pass
    else:
        suppl = Chem.SDMolSupplier(orig_ligand_path, removeHs=True)
        mol = [mol for mol in suppl][0]
        Chem.MolToPDBFile(mol, temp_ligand_path)

    try:
        lig_inter, rec_inter = binana.load_ligand_receptor.from_known(temp_ligand_path, data)

        cation_pi_inf = binana.interactions.get_cation_pi(lig_inter, rec_inter)  # label中顺序是配体在前,受体在后.counts中出现的是带电的
        halogen_bonds_inf = binana.interactions.get_halogen_bonds(lig_inter, rec_inter)  # label中出现的名字表示供体
        hydrogen_bonds_inf = binana.interactions.get_hydrogen_bonds(lig_inter, rec_inter)  # label中出现的名字表示供体,顺序依然是配体在前受体在后
        pi_pi_inf = binana.interactions.get_pi_pi(lig_inter, rec_inter)  # 配体环-蛋白环(Pi堆积) T堆积类似,不做区分edge和face
        inter_list = [cation_pi_inf['labels'], halogen_bonds_inf['labels'], hydrogen_bonds_inf['labels'], pi_pi_inf['labels']]
        prompt = get_interaction_prompt("UNL", inter_list)

        return prompt
    except Exception as e:
        print(e)
        return None


def set_prompt_value(inter_items, inter_index, lig_name, prompt):
    # 对prompt进行赋值
    for inter_item in inter_items:
        lig_index = []
        pro_index = []
        for idx, inter in enumerate(inter_item):
            if isinstance(inter, str) and inter != 'LIGAND' and inter != 'RECEPTOR':
                if '/' in inter:
                    inter_atoms = inter.split('/')
                    for inter_atom in inter_atoms:
                        inter_atom = inter_atom.split(':')
                        res_name = re.findall(r'(\w+)\([-]?\d+\)', inter_atom[1])[0]
                        atom_index = int(re.findall(r'\((\d+)\)', inter_atom[2])[0])
                        if res_name == lig_name:
                            lig_index.append(atom_index - 1)
                        else:
                            pro_index.append(atom_index)

                else:
                    inter_atom = inter.split(':')
                    res_name = re.findall(r'(\w+)\([-]?\d+\)', inter_atom[1])[0]
                    atom_index = int(re.findall(r'\((\d+)\)', inter_atom[2])[0])
                    if res_name == lig_name:
                        lig_index.append(atom_index - 1)
                    else:
                        pro_index.append(atom_index)
        lig_index = list(set(lig_index))
        pro_index = list(set(pro_index))
        prompt[inter_index].append((lig_index, pro_index))


# 该函数返回相互作用提示,提示格式为样本中参与相互作用的配体原子序号,蛋白原子序号,相互作用类型
def get_interaction_prompt(lig_name, inter_list):
    prompt = {0: [], 1: [], 2: [], 3: []}
    for index, inters in enumerate(inter_list):
        if inters == []:
            continue
        if index == 3:
            for inter in inters.values():
                if inter == []:
                    continue
                set_prompt_value(inter, index, lig_name, prompt)
        else:
            set_prompt_value(inters, index, lig_name, prompt)

    return prompt


def get_prompt(key, args):
    db = lmdb.open(
        args.source_db_path,
        map_size=15 * (1024 * 1024 * 1024),
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    db.begin().get(key)
    data = pickle.loads(db.begin().get(key))
    db.close()
    # if len(data['protein_element'])<100:
    # print('found small context {}'.format(len(data['protein_element'])))
    if any(i in data['ligand_element'] for i in [1, 12, 13, 14, 21, 23, 24, 26, 34, 42, 44, 50, 79, 80]):
        return None
    part_data = {
        'protein_element': data['protein_element'],
        'protein_atom_to_aa_type': data['protein_atom_to_aa_type'],
        'protein_atom_name': data['protein_atom_name'],
        'protein_res_id': data['protein_res_id'],
        'protein_pos': data['protein_pos'],
        'protein_filename': data['protein_filename'],

        'ligand_filename': data['ligand_filename'],
        'ligand_pos': data['ligand_pos'],
    }
    prompts = inter_detect(part_data, args)
    return key, prompts


# 统计相互作用的一些函数
# 提取残基类型
def get_res(x, data):
    res = []
    for i in x:
        res.append(AA_NUMBER[data[i]])
    return res


# 提取配体原子类型
def get_lig_atom(x, data):
    atom = []
    for i in x:
        atom.append(LIG_ATOM[data[i].item()])
    return atom


# 提取蛋白原子类型
def get_pro_atom(x, data):
    atom = []
    for i in x:
        atom.append(data[i])
    return atom


if __name__ == "__main__":
    set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data_path', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--source_db_path', type=str, default='../interdiff_data/pocket.lmdb')
    parser.add_argument('--temp_path', type=str, default='../interdiff_data/temp')
    parser.add_argument('--save_db_path', type=str, default='../interdiff_data/pocket_with_prompt.lmdb')
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.temp_path, exist_ok=True)

    db = lmdb.open(
        args.source_db_path,
        map_size=15 * (1024 * 1024 * 1024),  # 5GB
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    db2 = lmdb.open(
        args.save_db_path,
        map_size=15 * (1024 * 1024 * 1024),
        create=True,
        subdir=False,
        readonly=False,  # Writable
        lock=False,
        readahead=False,
        meminit=False,
    )
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    print(len(keys))

    inter_dis = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}
    inter = {0: 'caption', 1: 'halogen', 2: 'hydrogen', 3: 'pi'}

    pool = Pool(processes=args.num_workers)
    results = pool.imap_unordered(partial(get_prompt, args=args), keys)
    for prompt in tqdm(results, total=len(keys)):
        if prompt is None:
            continue
        key, prompt = prompt
        with db2.begin(write=True, buffers=True) as txn:
            no_inters = 0
            data = pickle.loads(db.begin().get(key))
            if prompt is None:
                continue
            else:
                num_inters = 0
                for _, v in prompt.items():
                    num_inters += len(v)
                if num_inters == 0:
                    no_inters += 1
                    continue
                data['prompt'] = prompt

                # remove the interaction involves more than one residue
                temp_prompt = {}
                for prompt_id, value in prompt.items():
                    temp_prompt[prompt_id] = set()
                    for sub_prompt in value:
                        protein_atoms = sub_prompt[1]
                        res_ids = set(data['protein_res_id'][protein_atoms])
                        if len(res_ids) == 1:
                            temp_prompt[prompt_id] = temp_prompt[prompt_id].union(res_ids)
                        else:
                            print('detect one interaction involves more than one residues')

                # remove the residue involves more than one interaction
                temp, del_res_ids = set(), set()
                for prompt_id, res_ids in temp_prompt.items():
                    # detect if any residues with more than one interaction
                    repeat_res_id = temp.intersection(res_ids)
                    del_res_ids.union(repeat_res_id)
                    temp.union(res_ids)

                for prompt_id, value in temp_prompt.items():
                    value -= del_res_ids

                # add the interaction prompt to every protein atom
                interactions = np.zeros(len(data['protein_element']), dtype=np.int64)
                for prompt_id, res_ids in temp_prompt.items():
                    atom_indexes = np.isin(data['protein_res_id'], list(res_ids))
                    aminos = data['protein_atom_to_aa_type']

                    all_interaction_prompt = np.array([
                        interaction_prompt.get(AA_NUMBER[amino] + '_' + INTERACTION[prompt_id], 44) for amino in aminos
                    ])[atom_indexes]
                    interactions[atom_indexes] = all_interaction_prompt

                data['interaction'] = interactions

                torchify_data = torchify_dict(data)
                txn.put(key=key, value=pickle.dumps(torchify_data))

                # add to csv
                inter_dis[0].append(len(prompt[0]))  # 记录该相互作用数目
                inter_dis[1].append(len(prompt[1]))  # 记录该相互作用数目
                inter_dis[2].append(len(prompt[2]))  # 记录该相互作用数目
                inter_dis[3].append(len(prompt[3]))  # 记录该相互作用数目
                inter_dis[4].append(len(data['ligand_element']))  # 记录复合物配体原子数
                inter_dis[5].append(len(data['protein_element']))  # 记录复合物蛋白口袋原子数
                interactions = []
                for idx, interaction in prompt.items():
                    for j in interaction:
                        if j != []:
                            temp = {
                                INTERACTION[idx]: (
                                    (
                                        j,
                                        get_lig_atom(j[0], data['ligand_element']),
                                        get_pro_atom(j[1], data['protein_atom_name']),
                                        get_res(j[1], data['protein_atom_to_aa_type']),
                                        data['protein_res_id'][j[1]].tolist()
                                    )
                                )
                            }
                            interactions.append(temp)

                inter_dis[6].append(interactions)
                inter_dis[7].append(data['ligand_smiles'])
                inter_dis[8].append(data['ligand_filename'])
    print(no_inters)
    dataframe = pd.DataFrame({
        'cation_pi_inf': inter_dis[0],
        'halogen_bonds_inf': inter_dis[1],
        'hydrogen_bonds_inf': inter_dis[2],
        'pi-pi': inter_dis[3],
        'lig_length': inter_dis[4],
        'pock_length': inter_dis[5],
        'interactions': inter_dis[6],
        'smiles': inter_dis[7],
        'ligand_file': inter_dis[8]})
    dataframe.to_csv("data/statistics11.csv", index=False, sep=',')

    # split train/test data
    with db2.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))

    train = random.sample(range(len(keys)), int(0.995 * len(keys)))
    test = list(set(range(len(keys))).difference(set(train)))
    print('train nums: {} , test nums {}'.format(len(train), len(test)))
    torch.save({'train': train, 'test': test}, '../interdiff_data/prompt_split.pt')
