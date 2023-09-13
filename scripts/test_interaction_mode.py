import argparse, pickle
import lmdb, copy
import shutil
import torch
# from autodock import DockVina_smi
import pandas as pd
from torch.utils.data import Dataset
from rdkit import Chem
from tqdm import tqdm
from models.new_score_model import ScorePosNet3D
from scripts_orig.sample_diffusion import sample_diffusion_ligand
from models.add_finetune import BottleneckModel
from utils import misc, reconstruct, transforms
from collections import Counter
import numpy as np
from collections import OrderedDict as od
from pathlib import Path, PurePath
import random
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
    'ALA_halogen': 0, 'ALA_hydrogen': 1, 'ASP_halogen': 2, 'ASP_hydrogen': 3, 'GLU_halogen': 4,
    'GLU_hydrogen': 5,
    'CYS_halogen': 6, 'CYS_hydrogen': 7, 'GLY_halogen': 8, 'GLY_hydrogen': 9, 'HIS_halogen': 10,
    'HIS_hydrogen': 11,
    'ILE_halogen': 12, 'ILE_hydrogen': 13, 'LYS_halogen': 14, 'LYS_hydrogen': 15, 'LEU_halogen': 16,
    'LEU_hydrogen': 17,
    'MET_halogen': 18, 'MET_hydrogen': 19, 'ASN_halogen': 20, 'ASN_hydrogen': 21, 'PRO_halogen': 22,
    'PRO_hydrogen': 23,
    'GLN_halogen': 24, 'GLN_hydrogen': 25, 'ARG_halogen': 26, 'ARG_hydrogen': 27, 'SER_halogen': 28,
    'SER_hydrogen': 29,
    'THR_halogen': 30, 'THR_hydrogen': 31, 'VAL_halogen': 32, 'VAL_hydrogen': 33, 'TRP_halogen': 34,
    'TRP_hydrogen': 35,
    'TYR_caption': 36, 'TYR_halogen': 37, 'TYR_hydrogen': 38, 'TYR_pi': 39, 'PHE_caption': 40,
    'PHE_halogen': 41, 'PHE_hydrogen': 42, 'PHE_pi': 43,
    'None': 44
}

PROMPT_TYPE = {
    i + 1: k for i, (k, _) in enumerate(PROMPT.items())
}

used_keys = ['protein_pos', 'ligand_pos', 'protein_atom_feature', 'ligand_atom_feature_full',
             'protein_element', 'ligand_element']


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
            data[i + '_batch'] = torch.tensor(
                sum([[idx] * len(j[i]) for idx, j in enumerate(batch)], []))
            data[i + '_length'] = [len(j[i]) for j in batch]
        else:
            data[i] = torch.cat([j[i] for j in batch])

    return data, pro_data


# 评估测试结果
def evaluation_sampling(result_path):
    results = torch.load(result_path)  # 测试的结果,总生成样本数为30*6*4=720
    original = torch.load('test_data.pt')  # 原始的30个测试样本,用于提供未修改的提示和蛋白的信息
    # original = [i for idx,i in enumerate(original) if idx !=0 and idx != 1]
    num_samples = 0

    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    # vina = DockVina_smi()
    all_pair_dist, all_bond_dist, mols, mols_prop = [], [], [], []
    id_list = []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    data = {'protein': [], 'ligand': [], 'smiles': [], 'res_neighbor': [], 'new_inter': [],
            'generate_inter': [], 'docking_score': []}
    inters_set = []  # 保存生成样本的全部提示
    # 生成时batchsize为6,result大小为720/6=120,循环120次

    for example_idx, r in enumerate(tqdm(results, desc='Eval')):
        batch_data = results[r]
        samples = batch_data['samples']
        protein = batch_data['data']
        for k in tqdm(range(len(samples)), total=len(samples)):
            all_pred_ligand_pos = samples[k][
                'pred_ligand_pos']  # [num_samples, num_steps, num_atoms, 3]
            all_pred_ligand_v = samples[k]['pred_ligand_v']
            num_samples += len(all_pred_ligand_pos)

            # batchsize是10,因此下面循环为10次
            for sample_idx, (pred_pos, pred_v) in enumerate(
                    zip(all_pred_ligand_pos, all_pred_ligand_v)):
                used_protein = {
                    'protein_res_id': protein['protein_res_id'][sample_idx],
                    'protein_atom_to_aa_type': protein['protein_atom_to_aa_type'][sample_idx],
                    'protein_atom_name': protein['protein_atom_name'][sample_idx],
                    'protein_element': protein['protein_element'][sample_idx],
                    'protein_pos': protein['protein_pos'][sample_idx]
                }

                # stability check
                pred_atom_type = transforms.get_atomic_number_from_index(pred_v,
                                                                         mode='add_aromatic')
                all_atom_types += Counter(pred_atom_type)
                # r_stable = analyze.check_stability(pred_pos, pred_atom_type)
                # all_mol_stable += r_stable[0]
                # all_atom_stable += r_stable[1]
                # all_n_atom += r_stable[2]
                # pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
                # all_pair_dist += pair_dist
                # 总样本的索引
                idx = example_idx * 300 + k * 10 + sample_idx
                # idx = example_idx*15+sample_idx
                try:
                    ori = original[example_idx * 10 + sample_idx]
                    # ori = original[1]
                    # data['res_neighbor'].append(ori['res_neighbor'])
                    # data['new_inter'].append(get_inter(ori))
                    data['ligand'].append('{}.pdb'.format(idx))
                    data['protein'].append(ori['protein_filename'])

                    pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode='add_aromatic')

                    mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type,
                                                                 pred_aromatic)

                    smiles = Chem.MolToSmiles(mol)

                    data['smiles'].append(smiles)
                    """
                    if '.' not in smiles:
                        n_complete += 1
                        id_list.append(idx)
                        mols.append(mol)
                        chem_results = scoring_func.get_chem(mol)
                        mols_prop.append(chem_results)
                        n_eval_success += 1
                        bond_dist = eval_bond_length.bond_distance_from_mol(mol)
                        all_bond_dist += bond_dist

                        success_pair_dist += pair_dist
                        success_atom_types += Counter(pred_atom_type)

                    
                    
                    Chem.MolToPDBFile(mol,"eval_results/{}.pdb".format(idx))

                    lig_inter, rec_inter = binana.load_ligand_receptor.from_known("eval_results/{}.pdb".format(idx),used_protein)
                    
                    cation_pi_inf = binana.interactions.get_cation_pi(lig_inter, rec_inter)#label中顺序是配体在前,受体在后.counts中出现的是带电的
                    halogen_bonds_inf = binana.interactions.get_halogen_bonds(lig_inter, rec_inter)#label中出现的名字表示供体
                    hydrogen_bonds_inf = binana.interactions.get_hydrogen_bonds(lig_inter, rec_inter)#label中出现的名字表示供体,顺序依然是配体在前受体在后
                    pi_pi_inf = binana.interactions.get_pi_pi(lig_inter, rec_inter)#配体环-蛋白环(Pi堆积) T堆积类似,不做区分edge和face
                    inter_list = [cation_pi_inf['labels'],halogen_bonds_inf['labels'],hydrogen_bonds_inf['labels'],pi_pi_inf['labels']]#残基从1开始计数,原子序号从0计数
                    prompt = get_interaction_prompt("UNL",inter_list)
                    inters_set.append([inter_list,prompt])
                    data['generate_inter'].append(get_inter_from_prompt(prompt,ori))
                    """
                except Exception as e:
                    print(e)
                    data['smiles'].append([])
                    # traceback.print_exc()
                    prompt = ['fail']
                    data['generate_inter'].append(['fail'])

                    inters_set.append(prompt)
                    logger.warning('Reconstruct failed %s' % f'{example_idx}_{sample_idx}')
                    continue
                n_recon_success += 1
    """
    fraction_mol_stable = all_mol_stable / num_samples
    fraction_atm_stable = all_atom_stable / all_n_atom
    fraction_recon = n_recon_success / num_samples
    fraction_eval = n_eval_success / num_samples
    fraction_complete = n_complete / num_samples
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }
    # assert len(inters_set) == 10000
    # assert len(inters_set) == 1000
    torch.save({'mols':mols,
                'id_list':id_list,
                'mol_prop':mols_prop,
                'success_pair_dist':success_pair_dist,
                'all_bond_dist':all_bond_dist,
                'success_atom_types':success_atom_types,
                'valid_stability':validity_dict,
                'inters':inters_set
                },'new_equivar_randomatoms.pt')
    # torch.save(inters_set,"new_equivar_randomatoms.pt")
    """
    dataframe = pd.DataFrame({'protein_file': data['protein'], 'ligand_file': data['ligand'],
                              'smiles': data['smiles']})
    #   'res_neighbor':data['res_neighbor'],
    # 'new_inter':data['new_inter'],'generate_inter':data['generate_inter']} )
    dataframe.to_csv("./new_equivar_randomatoms.csv", index=False, sep=',')

    return None


def get_testset():
    db = lmdb.open(
        "final_input_filter_rmrepeat111.lmdb",
        map_size=15 * (1024 * 1024 * 1024),  # 10GB
        create=False,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
    )
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))
    train = random.sample(range(len(keys)), 6000)
    test_data = []
    count = 0
    # 抽取30个样本做测试,确保TYR和PHE两个残基的数目至少有一个大于三,并且不存在相互作用
    # 因为仅TYR与PHE两个残基存在4种相互作用
    for i in train:
        if count == 30:
            break
        key = keys[i]
        data = pickle.loads(db.begin().get(key))
        PHE_idx = np.argwhere(data['protein_atom_to_aa_type'] == 4)
        TYR_idx = np.argwhere(data['protein_atom_to_aa_type'] == 19)
        PHE_num = np.unique(data['protein_res_id'][PHE_idx])
        TYR_num = np.unique(data['protein_res_id'][TYR_idx])

        PHE_has_nointer = 0
        for i in range(len(PHE_num)):
            idx = np.argwhere(data['protein_res_id'] == PHE_num[i]).squeeze()
            if sum(data['interaction'][idx]).item() == 0:
                PHE_has_nointer = PHE_has_nointer + 1
        TYR_has_nointer = 0
        for i in range(len(TYR_num)):
            idx = np.argwhere(data['protein_res_id'] == TYR_num[i]).squeeze()
            if sum(data['interaction'][idx]).item() == 0:
                TYR_has_nointer = TYR_has_nointer + 1
        # 符合条件则选为测试样本
        if PHE_has_nointer >= 3 or TYR_has_nointer >= 3:
            # if data['protein_atom_to_aa_type']
            test_data.append(data)
            count = count + 1
    db.close()
    print('get {} data'.format(count))
    test_data = add_prompt(test_data)
    torch.save(test_data, "test_prompt.pt")
    return test_data


def add_prompt(data):
    added_data = {
        0: [], 1: [], 2: [], 3: [], 5: data
    }
    # 对每个相互作用,每个测试样本给6个不同的提示,共测试4*6*30=720个样本
    for j in range(4):
        for i in range(len(data)):
            used = data[i]
            replace = copy.deepcopy(used)
            added = copy.deepcopy(used)
            replace2 = copy.deepcopy(used)
            added2 = copy.deepcopy(used)
            replace3 = copy.deepcopy(used)
            added3 = copy.deepcopy(used)

            inter_idx = torch.argwhere(used['interaction'] != 0)
            inter_res = np.unique(used['protein_res_id'][inter_idx])
            no_inter_idx = torch.argwhere(used['interaction'] == 0)
            no_inter_res = used['protein_res_id'][no_inter_idx]
            if j == 0 or j == 3:
                no_inter_idx = np.argwhere(np.logical_and(used['interaction'] == 0, np.logical_or(
                    used['protein_atom_to_aa_type'] == 4, used['protein_atom_to_aa_type'] == 19)))
                no_inter_res = used['protein_res_id'][no_inter_idx]
            sample_inter = random.sample(no_inter_res.squeeze().tolist(), 3)
            if len(inter_res) == 1:
                kill_inter = random.sample([inter_res.squeeze().tolist()], 1)
            else:
                kill_inter = random.sample(inter_res.squeeze().tolist(), 1)
            # 从现有的相互作用中替换一个相互作用提示,测试模型替换相互作用的能力,对每个样本重复三次
            replace_res_idx = np.argwhere(used['protein_res_id'] == sample_inter[0])
            amino = AA_NUMBER[np.unique(used['protein_atom_to_aa_type'][replace_res_idx]).item()]
            new_key = amino + '_' + INTERACTION[j]
            replace['interaction'][replace_res_idx.squeeze()] = PROMPT[new_key] + 1
            replace['interaction'][
                np.argwhere(used['protein_res_id'] == kill_inter[0]).squeeze()] = 0

            replace_res_idx = np.argwhere(used['protein_res_id'] == sample_inter[1])
            amino = AA_NUMBER[np.unique(used['protein_atom_to_aa_type'][replace_res_idx]).item()]
            new_key = amino + '_' + INTERACTION[j]
            replace2['interaction'][replace_res_idx.squeeze()] = PROMPT[new_key] + 1
            replace2['interaction'][
                np.argwhere(used['protein_res_id'] == kill_inter[0]).squeeze()] = 0

            replace_res_idx = np.argwhere(used['protein_res_id'] == sample_inter[2])
            amino = AA_NUMBER[np.unique(used['protein_atom_to_aa_type'][replace_res_idx]).item()]
            new_key = amino + '_' + INTERACTION[j]
            replace3['interaction'][replace_res_idx.squeeze()] = PROMPT[new_key] + 1
            replace3['interaction'][
                np.argwhere(used['protein_res_id'] == kill_inter[0]).squeeze()] = 0
            ##增加一个相互作用提示,测试模型添加相互作用的能力,对每个样本重复三次
            add_res_idx = np.argwhere(used['protein_res_id'] == sample_inter[0]).squeeze()
            amino = AA_NUMBER[np.unique(used['protein_atom_to_aa_type'][add_res_idx]).item()]
            new_key = amino + '_' + INTERACTION[j]
            added['interaction'][add_res_idx] = PROMPT[new_key] + 1

            add_res_idx = np.argwhere(used['protein_res_id'] == sample_inter[1]).squeeze()
            amino = AA_NUMBER[np.unique(used['protein_atom_to_aa_type'][add_res_idx]).item()]
            new_key = amino + '_' + INTERACTION[j]
            added2['interaction'][add_res_idx] = PROMPT[new_key] + 1

            add_res_idx = np.argwhere(used['protein_res_id'] == sample_inter[2]).squeeze()
            amino = AA_NUMBER[np.unique(used['protein_atom_to_aa_type'][add_res_idx]).item()]
            new_key = amino + '_' + INTERACTION[j]
            added3['interaction'][add_res_idx] = PROMPT[new_key] + 1
            # 对每个相互作用,每个样本用不同相互作用提示测试6次
            added_data[j].append([replace, replace2, replace3, added, added2, added3])

    return added_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/training.yml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--result_path', type=str, default='./outputs_systems')
    parser.add_argument('--evaluation', type=bool, default=True)
    parser.add_argument('--evaluation_data', type=str, default='test_prompt.pt')
    parser.add_argument('--checkpoint_date', type=str, default='training_2023_07_11__5emb')
    parser.add_argument('--save_name', type=str, default='equari.pt')
    parser.add_argument('--use_four_emb', type=bool, default=True)

    args = parser.parse_args()
    if Path(args.evaluation_data).is_file():
        test_data = torch.load(args.evaluation_data)  # [50:]
    # test_data = [i for idx,i in enumerate(test_data) if idx !=0 and idx != 1]
    else:
        test_data = get_testset()
    logger = misc.get_logger('evaluate')

    # generate = sum([sum(test_data[i],[])for i in range(4)],[])
    # Load config
    config = misc.load_config(args.config)
    logger.info(config)


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

    # if args.evaluation:
    if False:
        evaluation_sampling(PurePath.joinpath(Path(args.result_path), Path(args.save_name)))
    else:
        # size_sampler = torch.load('atom_size_sampler.pt')['bins']
        test_data = PocketLigandPairDataset(test_data)
        test_loader = torch.utils.data.DataLoader(test_data, args.batch_size, shuffle=False,
                                                  collate_fn=collate_fun)
        config = misc.load_config(args.config)
        # Load model
        model = ScorePosNet3D(
            config.model,
            protein_atom_feature_dim=25,
            ligand_atom_feature_dim=15,
            use_four_emb=args.use_four_emb,
            device=args.device
        )

        model = BottleneckModel(model).to(args.device)

        dict_buffer = torch.load('training_2023_07_11__45emb/checkpoints/5640.pt',
                                 map_location=lambda storage, loc: storage.cuda(0))
        re_dict = od()
        for k, v in dict_buffer['model'].items():
            re_dict[k[10:]] = v
        model.load_state_dict(re_dict, strict=False)
        results = {}
        repeat_times = 30

        for idx, data in enumerate(test_loader):

            data, pro_data = data
            results[idx] = {'data': pro_data, 'samples': []}
            pocket_size = [atom_num.get_space_size(i) for i in pro_data['protein_pos']]
            for k in range(repeat_times):
                print('sampling {} batch and {} repeat time'.format(idx, k))
                pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list, atts = sample_diffusion_ligand(
                    model, data, args.batch_size,
                    batch_size=args.batch_size, device=args.device,
                    num_steps=1000,
                    pos_only=False,
                    center_pos_mode='protein',
                    sample_num_atoms='prior',
                    return_attention=False,
                    # size_sampler = size_sampler,
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

        logger.info('Sample done!')

        result_path = args.result_path
        Path.mkdir(Path(result_path), exist_ok=True)
        shutil.copyfile(args.config, PurePath.joinpath(Path(result_path), Path('sample.yml')))
        torch.save(results, PurePath.joinpath(Path(result_path), Path(args.save_name)))
