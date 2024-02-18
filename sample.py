import argparse, pickle, binana
import lmdb, random, copy
import shutil
import torch, os
from collections import defaultdict
import pandas as pd
from torch.utils.data import Dataset
from extract_dataset_moad import get_interaction_prompt
from rdkit import Chem
from tqdm import tqdm
import utils.misc as misc
from models.new_score_model2 import ScorePosNet3D
from sample_diffusion import sample_diffusion_ligand
from add_finetune import BottleneckModel
from utils import misc, reconstruct, transforms
from collections import Counter
from utils.evaluation import analyze, eval_bond_length, scoring_func
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict as od
from pathlib import Path, PurePath
import random, traceback
from utils.evaluation import atom_num

# kk = torch.load('test_inters.pt')
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
INTERACTION = {
    0: 'caption', 1: 'halogen', 2: 'hydrogen', 3: 'pi'
}

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

PROMPT_TYPE = {
    i + 1: k for i, (k, _) in enumerate(PROMPT.items())
}

used_keys = ['protein_pos', 'ligand_pos', 'protein_atom_feature', 'ligand_atom_feature_full', 'protein_element', 'ligand_element']


def get_inter(data):
    ret = []
    res_hasinter = np.argwhere(data['interaction'] != 0).squeeze()
    res_id = list(set(data['protein_res_id'][res_hasinter].tolist()))
    for k in res_id:
        idx = np.argwhere(data['protein_res_id'] == k).squeeze()
        amino = AA_NUMBER[data['protein_atom_to_aa_type'][idx][0]]
        inter_type = PROMPT_TYPE[data['interaction'][idx][0].item()]
        inter_type = inter_type.split('_')[1]
        ret.append((amino, k, inter_type))
    return ret


def get_inter_from_prompt(prompt, original):
    ret = []
    if prompt == ['fail']:
        return prompt
    # prompt = prompt[1]
    if prompt == []:
        return ret
    try:
        for key, value in prompt.items():
            if value != []:
                for j in value:
                    res_id = original['protein_res_id'][j[1]]  # 确认这里的从0计数还是从1计数：从0开始计数
                    res_id = list(set(res_id))
                    amino_id = original['protein_atom_to_aa_type'][j[1]].tolist()
                    # assert len(set(amino_id)) == 1, 'found one interaction involves more than one residue'
                    if len(set(res_id)) > 1:
                        amino_id = list(set(amino_id))
                        for k in range(len(amino_id)):
                            amino = AA_NUMBER[amino_id[k]]
                            gene = (amino, res_id[k], INTERACTION[key])
                            if gene not in ret:
                                ret.append((amino, res_id[k], INTERACTION[key]))
                    else:
                        amino = AA_NUMBER[amino_id[0]]
                        gene = (amino, res_id[0], INTERACTION[key])
                        if gene not in ret:
                            ret.append((amino, res_id[0], INTERACTION[key]))
        return ret
    except:
        return ['error']


def collate_fun(batch):
    data = {}
    pro_data = {}
    pro_data['protein_res_id'] = [j['protein_res_id'] for j in batch]
    pro_data['protein_atom_to_aa_type'] = [j['protein_atom_to_aa_type'] for j in batch]
    pro_data['protein_atom_name'] = [j['protein_atom_name'] for j in batch]
    pro_data['protein_element'] = [j['protein_element'] for j in batch]
    pro_data['protein_pos'] = [j['protein_pos'].numpy() for j in batch]
    data['interaction'] = torch.cat([j['interaction'] for j in batch]).squeeze()
    for i in used_keys:
        if 'element' in i:
            data[i + '_batch'] = torch.tensor(sum([[idx] * len(j[i]) for idx, j in enumerate(batch)], []))
            data[i + '_length'] = [len(j[i]) for j in batch]
        else:
            data[i] = torch.cat([j[i] for j in batch])

    return data, pro_data


# evaluat sampling results
def evaluation_sampling(result_path):
    results = torch.load(result_path)  
    original = torch.load('data/test_data.pt')  
    # original = [i for idx,i in enumerate(original) if idx !=0 and idx != 1]
    num_samples = 0


    n_recon_success = 0

    all_pair_dist, all_bond_dist, mols, mols_prop = [], [], [], []

    all_atom_types = Counter()


    data = {'protein': [], 'ligand': [], 'smiles': [], 'res_neighbor': [], 'ori_inter': [], 'generate_inter': [], 'docking_score': []}
    inters_set = []

    for example_idx, r in enumerate(tqdm(results, desc='Eval')):  #batch numbers
        batch_data = results[r]
        samples = batch_data['samples']
        protein = batch_data['data']
        for k in tqdm(range(len(samples)), total=len(samples)):  # repeat times for each batch
            all_pred_ligand_pos = samples[k]['pred_ligand_pos']  # [num_samples, num_steps, num_atoms, 3]
            all_pred_ligand_v = samples[k]['pred_ligand_v']
            num_samples += len(all_pred_ligand_pos)

            # batchsize(number of protein targets in each batch)
            for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_ligand_pos, all_pred_ligand_v)):
                used_protein = {
                    'protein_res_id': protein['protein_res_id'][sample_idx],
                    'protein_atom_to_aa_type': protein['protein_atom_to_aa_type'][sample_idx],
                    'protein_atom_name': protein['protein_atom_name'][sample_idx],
                    'protein_element': protein['protein_element'][sample_idx],
                    'protein_pos': protein['protein_pos'][sample_idx]
                }
                # predict the atom types
                pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode='add_aromatic')
                all_atom_types += Counter(pred_atom_type)

                pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
                all_pair_dist += pair_dist

                idx = example_idx * 1000 + k * 10 + sample_idx

                try:
                    ori = original[50 + sample_idx]

                    data['ori_inter'].append(get_inter(ori))
                    data['ligand'].append('{}.pdb'.format(idx))
                    data['protein'].append(ori['protein_filename'])

                    pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode='add_aromatic')

                    mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic, basic_mode=False)

                    smiles = Chem.MolToSmiles(mol)
                    data['smiles'].append(smiles)

                    if not os.path.exists("outputs_system/eval_results/{}.pdb".format(idx)):
                        os.makedirs("outputs_system/eval_results/", exist_ok=True)
                        Chem.MolToPDBFile(mol, "outputs_system/eval_results/{}.pdb".format(idx))

                    lig_inter, rec_inter = binana.load_ligand_receptor.from_known("outputs_system/eval_results/{}.pdb".format(idx), used_protein)

                    cation_pi_inf = binana.interactions.get_cation_pi(lig_inter, rec_inter)  # label中顺序是配体在前,受体在后.counts中出现的是带电的
                    halogen_bonds_inf = binana.interactions.get_halogen_bonds(lig_inter, rec_inter)  # label中出现的名字表示供体
                    hydrogen_bonds_inf = binana.interactions.get_hydrogen_bonds(lig_inter, rec_inter)  # label中出现的名字表示供体,顺序依然是配体在前受体在后
                    pi_pi_inf = binana.interactions.get_pi_pi(lig_inter, rec_inter)  # 配体环-蛋白环(Pi堆积) T堆积类似,不做区分edge和face
                    inter_list = [cation_pi_inf['labels'], halogen_bonds_inf['labels'], hydrogen_bonds_inf['labels'], pi_pi_inf['labels']]  # 残基从1开始计数,原子序号从0计数
                    prompt = get_interaction_prompt("UNL", inter_list)
                    data['generate_inter'].append(get_inter_from_prompt(prompt, ori))

                except Exception as e:
                    print(e)
                    data['smiles'].append([])
                    # traceback.print_exc()
                    prompt = ['fail']
                    data['generate_inter'].append(['fail'])
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                    continue
                n_recon_success += 1

    dataframe = pd.DataFrame({'protein_file': data['protein'], 'ligand_file': data['ligand'],
                              'smiles': data['smiles'],
                              'ori_inter': data['ori_inter'], 'generate_inter': data['generate_inter']})
    dataframe.to_csv("outputs_system/equa_aromatic.csv", index=False, sep=',')
    success_rate = [0 for i in range(len(data['smiles']))]
    data = pd.read_csv("outputs_system/equa_aromatic.csv")
    for idx, i in enumerate(data['generate_inter']):
        generate_inter = eval(str(i))
        docked_resid = [l[1] for l in generate_inter]
        ori_inter = eval(str(data['ori_inter'][idx]))
        res_id = [k[1] for k in ori_inter]
        shot = 0
        for j in ori_inter:
            if j in generate_inter:
                shot = shot + 1
        if len(ori_inter) != 0:
            rate = shot / len(ori_inter)
        else:
            rate = 0
        success_rate[idx] = rate

    data['success_rate'] = success_rate
    data.to_csv("outputs_system/equa_aromatic.csv", index=False)
    print(data)
    
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--result_path', type=str, default='./outputs_system')
    parser.add_argument('--evaluation', type=bool, default=False)
    parser.add_argument('--evaluation_data', type=str, default='data/test_data.pt')
    parser.add_argument('--checkpoint_date', type=str, default='pretrained')
    parser.add_argument('--save_name', type=str, default='equa_last.pt')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--use_four_emb', type=bool, default=True)

    args = parser.parse_args()

    test_data = torch.load(args.evaluation_data)[50:]

    logger = misc.get_logger('evaluate')

    # generate = sum([sum(test_data[i],[])for i in range(4)],[])
    # Load config
    config = misc.load_config(args.config)
    # logger.info(config)


    class PocketLigandPairDataset(Dataset):
        def __init__(self, test_data):
            super().__init__()

            self.db = test_data
            self.keys = None

        def __len__(self):
            return len(self.db)

        def __getitem__(self, idx):
            data = self.db[idx]
            # print(idx)
            return data


    # fix all random seed
    seed = 2023


    def fix_seed(fix, seed):
        if fix:
            torch.manual_seed(seed)  # 为CPU设置随机种子
            torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
            torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子
            np.random.seed(seed)  # Numpy module.
            random.seed(seed)  # Python random module.	
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


    fix_seed(False, seed)

    if args.evaluation:
        evaluation_sampling(PurePath.joinpath(Path(args.result_path), Path(args.save_name)))
        logger.info("ouput molecules file in \'outputs_system\\eval_results\' ")
    else:
        size_sampler = torch.load('data/atom_size_sampler.pt')['bins']
        test_data = PocketLigandPairDataset(test_data)
        test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False, collate_fn=collate_fun)
        config = misc.load_config(args.config)
        # Load model
        model = ScorePosNet3D(
            config.model,
            protein_atom_feature_dim=25,
            ligand_atom_feature_dim=12,
            use_four_emb=args.use_four_emb,
            device=args.device
        )

        model = BottleneckModel(model).to(args.device)

        dict_buffer = torch.load('./pretrained/checkpoints/time_dep_3540.pt'.format(args.checkpoint_date), map_location=lambda storage, loc: storage.cuda())
        re_dict = od()
        for k, v in dict_buffer['model'].items():
            re_dict[k[10:]] = v
        model.load_state_dict(re_dict, strict=False)
        results = {}
        repeat_times = 1

        for idx, data in enumerate(test_loader):

            data, pro_data = data
            results[idx] = {'data': pro_data, 'samples': []}
            pocket_size = [atom_num.get_space_size(i) for i in pro_data['protein_pos']]
            for k in range(repeat_times):
                print('sampling {} batch and {} repeat time'.format(idx, k))
                pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, atts = sample_diffusion_ligand(
                    model, data, args.batch_size,
                    batch_size=args.batch_size, device=args.device,
                    num_steps=args.steps,
                    pos_only=False,
                    center_pos_mode='protein',
                    sample_num_atoms='prior',
                    return_attention=False,
                    size_sampler=size_sampler,
                    pocket_size=pocket_size
                )

                result = {
                    'pred_ligand_pos': pred_pos,
                    'pred_ligand_v': pred_v,
                    # 'pred_ligand_pos_traj': pred_pos_traj,
                    # 'pred_ligand_v_traj': pred_v_traj,
                    # 'atts':atts
                }
                results[idx]['samples'].append(result)
            break
        logger.info('Sample done!')

        result_path = args.result_path
        Path.mkdir(Path(result_path), exist_ok=True)
        shutil.copyfile(args.config, PurePath.joinpath(Path(result_path), Path('sample.yml')))
        torch.save(results, PurePath.joinpath(Path(result_path), Path(args.save_name)))

        re_dict = od()
        for k, v in dict_buffer['model'].items():
            re_dict[k[10:]] = v
        model.load_state_dict(re_dict, strict=False)
        results = {}
        repeat_times = 1

        for idx, data in enumerate(test_loader):

            data, pro_data = data
            results[idx] = {'data': pro_data, 'samples': []}
            pocket_size = [atom_num.get_space_size(i) for i in pro_data['protein_pos']]
            for k in range(repeat_times):
                print('sampling {} batch and {} repeat time'.format(idx, k))
                pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, atts = sample_diffusion_ligand(
                    model, data, args.batch_size,
                    batch_size=args.batch_size, device=args.device,
                    num_steps=args.steps,
                    pos_only=False,
                    center_pos_mode='protein',
                    sample_num_atoms='prior',
                    return_attention=False,
                    size_sampler=size_sampler,
                    pocket_size=pocket_size
                )

                result = {
                    'pred_ligand_pos': pred_pos,
                    'pred_ligand_v': pred_v,
                    # 'pred_ligand_pos_traj': pred_pos_traj,
                    # 'pred_ligand_v_traj': pred_v_traj,
                    # 'atts':atts
                }
                results[idx]['samples'].append(result)
            break
        logger.info('Sample done!')

        result_path = args.result_path
        Path.mkdir(Path(result_path), exist_ok=True)
        shutil.copyfile(args.config, PurePath.joinpath(Path(result_path), Path('sample.yml')))
        torch.save(results, PurePath.joinpath(Path(result_path), Path(args.save_name)))