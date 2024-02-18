import os
import argparse, binana, re
import pickle
from rdkit import Chem
import lmdb, random, torch
from tqdm.auto import tqdm
from utils.data import PDBProtein, parse_sdf_file
from multiprocessing import Pool
import utils.transforms as trans
from pathlib import Path
import utils.misc as misc
import numpy as np
from torch_geometric.transforms import Compose


def remove_lines_containing_string(text_block, target_string):
    # 将文本块拆分成行
    lines = text_block.split('\n')

    # 使用列表推导式保留不包含目标字符串的行
    filtered_lines = [line for line in lines if target_string not in line]

    # 将过滤后的行重新组合成文本块
    result_text_block = '\n'.join(filtered_lines)

    return result_text_block


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
                    try:
                        res_name = re.findall(r'(\w+)\([-]?\d+\)', inter_atom[1])[0]
                    except:
                        res_name = re.findall(r'<(\d+)>\([-]?\d+\)', inter_atom[1])[0]
                    atom_index = int(re.findall(r'\((\d+)\)', inter_atom[2])[0])
                    if res_name == lig_name:
                        lig_index.append(atom_index - 1)
                    else:
                        pro_index.append(atom_index)
        lig_index = list(set(lig_index))
        pro_index = list(set(pro_index))
        prompt[inter_index].append((lig_index, pro_index))


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


def get_prompt(pdbfile, protein):
    lig_inter, rec_inter = binana.load_ligand_receptor.from_known(pdbfile, protein)

    cation_pi_inf = binana.interactions.get_cation_pi(lig_inter, rec_inter)  # label中顺序是配体在前,受体在后.counts中出现的是带电的
    halogen_bonds_inf = binana.interactions.get_halogen_bonds(lig_inter, rec_inter)  # label中出现的名字表示供体
    hydrogen_bonds_inf = binana.interactions.get_hydrogen_bonds(lig_inter, rec_inter)  # label中出现的名字表示供体,顺序依然是配体在前受体在后
    pi_pi_inf = binana.interactions.get_pi_pi(lig_inter, rec_inter)  # 配体环-蛋白环(Pi堆积) T堆积类似,不做区分edge和face
    inter_list = [cation_pi_inf['labels'], halogen_bonds_inf['labels'], hydrogen_bonds_inf['labels'], pi_pi_inf['labels']]  # 残基从1开始计数,原子序号从0计数
    prompt = get_interaction_prompt("UNL", inter_list)
    return prompt


AA_NAME_SYM = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
AA_NUMBER = {
    i: k for i, (k, _) in enumerate(AA_NAME_SYM.items())
}
LIG_ATOM = {
    5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 17: 'Cl', 15: 'P', 16: 'S', 35: 'Br', 53: 'I'
}
inter = {0: 'caption', 1: 'halogen', 2: 'hydrogen', 3: 'pi'}
PROMPT = {
    'ALA_halogen': 0, 'ALA_hydrogen': 1, 'ASP_halogen': 2, 'ASP_hydrogen': 3, 'GLU_halogen': 4, 'GLU_hydrogen': 5,
    'CYS_halogen': 6, 'CYS_hydrogen': 7, 'GLY_halogen': 8, 'GLY_hydrogen': 9, 'HIS_halogen': 10, 'HIS_hydrogen': 11,
    'ILE_halogen': 12, 'ILE_hydrogen': 13, 'LYS_halogen': 14, 'LYS_hydrogen': 15, 'LEU_halogen': 16, 'LEU_hydrogen': 17,
    'MET_halogen': 18, 'MET_hydrogen': 19, 'ASN_halogen': 20, 'ASN_hydrogen': 21, 'PRO_halogen': 22, 'PRO_hydrogen': 23,
    'GLN_halogen': 24, 'GLN_hydrogen': 25, 'ARG_halogen': 26, 'ARG_hydrogen': 27, 'SER_halogen': 28, 'SER_hydrogen': 29,
    'THR_halogen': 30, 'THR_hydrogen': 31, 'VAL_halogen': 32, 'VAL_hydrogen': 33, 'TRP_halogen': 34, 'TRP_hydrogen': 35,
    'TYR_caption': 36, 'TYR_halogen': 37, 'TYR_hydrogen': 38, 'TYR_pi': 39, 'PHE_caption': 40, 'PHE_halogen': 41, 'PHE_hydrogen': 42, 'PHE_pi': 43,
    'None': 44
}


def save_pocket(files, radius):
    file, file2 = files[0], files[1]  # pdb文件，小分子配体文件
    # 转换文件名，这里是crossdockced数据文件名处理
    # pro_basename = os.path.basename(file)[:10] + '.pdb'
    # pocket_fn = os.path.join(file.split('/')[1],pro_basename)
    # 读取蛋白文件
    with open(str(file), 'r') as f:
        pdb_block = f.read()
    remove_lines_containing_string(pdb_block, 'H0H')
    protein = PDBProtein(pdb_block)
    # 提取小分子特征
    ligand = parse_sdf_file(file2)
    # 根据配体提取口袋
    pros = protein.query_residues_ligand(ligand, radius)
    # 设置口袋名字
    # poc_name = pocket_fn.split('/')
    # 保存蛋白口袋为pdb文件
    pdb_block_pocket = protein.residues_to_pdb_block(pros)
    with open(os.path.join(str(file.parent), 'pocket', file.stem + 'pocket_{}'.format(radius) + '.pdb'), 'w') as f:
        f.write(pdb_block_pocket)
    # 口袋特征转换为模型需要的格式
    dicts = protein.to_residues(pros)
    return dicts, ligand


def get_interaction(data, prompt):
    interactions = torch.zeros(len(data['protein_element']), dtype=torch.int)
    for key, value in prompt.items():
        if value != []:
            for j in value:
                res_id = data['protein_res_id'][j[1]]
                res_id = list(set(res_id))
                if len(res_id) > 1:
                    raise RuntimeError('one reside has more than one interactions')
                res_index = np.where(data['protein_res_id'] == res_id)
                amino_id = data['protein_atom_to_aa_type'][j[1]].tolist()
                assert len(set(amino_id)) == 1, 'found one interaction involves more than one residue'
                amino = AA_NUMBER[amino_id[0]]
                new_key = amino + '_' + inter[key]
                if new_key == 'TRP_caption':
                    print('found abnormal')
                    new_key = 'TYR_caption'
                if new_key == 'TRP_pi':
                    print('found abnormal')
                    new_key = 'TYR_pi'
                if new_key == 'ARG_caption':
                    print('found abnormal')
                    new_key = 'TYR_caption'
                if new_key == 'HIS_pi':
                    print('found abnormal')
                    new_key = 'PHE_pi'
                if new_key == 'HIS_caption':
                    print('found abnormal')
                    new_key = 'PHE_caption'
                if new_key == 'LYS_caption':
                    print('found abnormal')
                    new_key = 'TYR_caption'
                interactions[res_index] = PROMPT[new_key] + 1

    return interactions


config = misc.load_config('./configs/training.yml')
protein_featurizer = trans.FeaturizeProteinAtom()
ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
transform_list = [
    protein_featurizer,
    ligand_featurizer,
]
transform = Compose(transform_list)


def func(i=None, data_path=None):
    try:
        # print(i.name.split('.pdb')[0] + '*.sdf')
        data_list = []
        for lig_file in list(data_path.glob(i.name.split('.pdb')[0] + '*.sdf')):
            pock_info, lig_data = save_pocket([i, str(lig_file)], args.radius)
            data = {}
            data['protein_element'] = pock_info['element']
            data['protein_pos'] = pock_info['pos']
            data['protein_filename'] = i.stem
            data['protein_atom_name'] = pock_info['atom_name']
            data['protein_atom_to_aa_type'] = pock_info['atom_to_aa_type']
            data['protein_is_backbone'] = pock_info['is_backbone']

            data['ligand_filename'] = str(lig_file)
            data['smiles'] = lig_data['smiles']
            data['ligand_element'] = lig_data['element']
            data['ligand_pos'] = lig_data['pos']
            data['ligand_atom_feature'] = lig_data['atom_feature']
            data['ligand_hybridization'] = lig_data['hybridization']
            data['protein_res_id'] = pock_info['res_id']
            data = transform(data)
            prompt = get_prompt(os.path.join(lig_file.parent, "sdf2pdb", lig_file.stem + '.pdb'), data)
            data['prompt'] = prompt
            interactions = get_interaction(data, prompt)
            data['interaction'] = interactions
            print((interactions > 0).sum())
            data_list.append(data)
        return data_list
    except Exception as e:
        return None
        print(e)


if __name__ == '__main__':
    import multiprocessing
    from multiprocessing import Pool
    from functools import partial
    from sklearn.model_selection import train_test_split

    parser = argparse.ArgumentParser()
    parser.add_argument('--radius', type=int, default=8)
    parser.add_argument('--config', type=str, default='./configs/training.yml')
    args = parser.parse_args()
    # test_path = Path('/data/wupeng/project/targetdiff-prop/targets/')
    ##SDF转pdb
    # used_sdf = Path('/data/wupeng/project/targetdiff-prop/targets/used_sdf')
    # for i in used_sdf.glob('*.sdf'):
    #     mol = Chem.SDMolSupplier(str(i))
    #     Chem.MolToPDBFile(mol[0], "./targets/used_sdf/{}.pdb".format(i.stem))

    config = misc.load_config(args.config)
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(config.data.transform.ligand_atom_mode)
    transform_list = [
        protein_featurizer,
        ligand_featurizer,
    ]
    transform = Compose(transform_list)

    # pdbqt_files = test_path.glob('*.pdbqt')
    # for i in pdbqt_files:
    # os.system('obabel -ipdbqt {} -opdb -d -O {}'.format(i,str(i.parent / i.stem)+'.pdb'))

    # 保存口袋信息
    # poc_file = test_path.glob('*.txt')
    # poc_extra = {}
    # for i in poc_file:
    #     name = i.stem
    #     with open(i, 'r') as f:
    #         poc_extra[name] = []
    #         for j in range(3):
    #             data = f.readline()
    #             poc_extra[name].append(float(data.strip().split()[2]))
    # torch.save(poc_extra, 'poc_extra.pt')

    # for i in pdb_files:
    # pdb = i.stem.split('_')[0]
    # pdb_file = i.parent / (pdb + '.pdb')
    # files.append((str(pdb_file),str(i)))

    # pdb_files = list(test_path.glob('*.pdb'))

    str_path = 'data/train'
    data_path = Path(str_path)
    os.makedirs(os.path.join(str_path, 'sdf2pdb'), exist_ok=True)
    os.makedirs(os.path.join(str_path, 'pocket'), exist_ok=True)
    # os.system('obabel -isdf {}/*.sdf -O {}/*.pdb'.format(str_path, os.path.join(str_path, 'sdf2pdb')))

    pdb_files = list(data_path.glob('*.pdb'))
    test_files = []
    env = lmdb.open('moad_data_train.lmdb', map_size=int(10 * 1024 * 1024 * 1024), subdir=False, lock=False)
    pool = Pool(30)
    with env.begin(write=True) as txn:
        for idx, data in enumerate(tqdm(pool.imap_unordered(partial(func, data_path=data_path), pdb_files), total=len(pdb_files))):
            if data:
                key = str(idx).encode('utf-8')
                value = pickle.dumps(data)
                txn.put(key, value)

        len_keys = len(list(txn.cursor().iternext(values=False)))
        train_data, test_data = train_test_split(range(len_keys), test_size=0.02, random_state=42)
        print(len(train_data), len(test_data))
        torch.save({'train': train_data, 'test': test_data}, "prompt_split_train.pt")
    env.close()
