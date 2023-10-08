import os, random, torch
import argparse
import multiprocessing as mp
import numpy as np
import pickle
import shutil
from functools import partial
import lmdb
from tqdm.auto import tqdm
from utils.data import PDBProtein, parse_sdf_file
from scripts.binana_script.detect_interactions import run_binana_command

AA_NAME_SYM = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H',
    'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q',
    'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y',
}
AA_NUMBER = {i: k for i, (k, _) in enumerate(AA_NAME_SYM.items())}
LIG_ATOM = {5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 17: 'Cl', 15: 'P', 16: 'S', 35: 'Br', 53: 'I'}

# INTERACTION = {0: 'cation_pi', 1: 'halogen', 2: 'hydrogen', 3: 'pi-pi'}
# INTERACTION = {0: 'cationPiInteractions', 1: 'halogenBonds', 2: 'hydrogenBonds', 3: 'piPiStackingInteractions', 4: 'saltBridges'}
INTERACTION = {'cationPiInteractions': 'cation_pi', 'halogenBonds': 'halogen', 'hydrogenBonds': 'hydrogen', 'piPiStackingInteractions': 'pi-pi'}

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


def load_item(item, path):
    pdb_path = os.path.join(path, item[0])
    sdf_path = os.path.join(path, item[1])
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    with open(sdf_path, 'r') as f:
        sdf_block = f.read()
    return pdb_block, sdf_block


def process_interactions(data):
    interactions = ['cationPiInteractions', 'halogenBonds', 'hydrogenBonds', 'piPiStackingInteractions', 'saltBridges']
    interactions = interactions[:-1]

    all_residue_ids = set()
    residues = {}
    for inter in interactions:
        for sub_inter in data[inter]:
            # inter keys: ligandAtoms, metrics, receptorAtoms
            res_ids_, res_names_ = [], []
            for res_atom in sub_inter['receptorAtoms']:
                res_ids_.append(res_atom['resID'])
                res_names_.append(res_atom['resName'])

            if len(np.unique(res_ids_)) == 1 and np.unique(res_ids_).item() not in all_residue_ids:
                res_id = np.unique(res_ids_).item()
                res_name = np.unique(res_names_).item()

                brief_inter = INTERACTION[inter]
                prompt_id = interaction_prompt.get(res_name + '_' + brief_inter, 44)
                residues[res_id] = prompt_id

                all_residue_ids.union({np.unique(res_ids_).item()})

    return residues


def process_item(item, args):
    i, item = item
    try:
        pdb_block, sdf_block = load_item(item, args.source_data_path)
        protein = PDBProtein(pdb_block)
        ligand_dict = parse_sdf_file(os.path.join(args.source_data_path, item[1]))

        selected_residues = protein.query_residues_ligand_(ligand_dict, args.radius)
        pdb_block_pocket = protein.residues_to_pdb_block(selected_residues)

        ligand_fn = item[1]
        pocket_fn = ligand_fn[:-4] + '_pocket%d.pdb' % args.radius
        ligand_dest = os.path.join(args.save_pocket_path, ligand_fn)
        pocket_dest = os.path.join(args.save_pocket_path, pocket_fn)
        os.makedirs(os.path.dirname(ligand_dest), exist_ok=True)

        shutil.copyfile(
            src=os.path.join(args.source_data_path, ligand_fn),
            dst=os.path.join(args.save_pocket_path, ligand_fn)
        )
        with open(pocket_dest, 'w') as f:
            f.write(pdb_block_pocket)

        # detect interaction
        interactions = run_binana_command((pocket_dest, ligand_dest, args.temp_path, str(i)))
        interactions = process_interactions(interactions)

        return (pocket_fn, ligand_fn, item[0], item[2]), (ligand_dict, selected_residues, interactions)  # item[0]: original protein filename; item[2]: rmsd.
    except Exception as e:
        return (None, item[1], item[0], item[2]), None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data_path', type=str, default='../interdiff_data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--save_pocket_path', type=str, default='../interdiff_data/crossdocked_v1.1_rmsd1.0_pocket_')
    parser.add_argument('--temp_path', type=str, default='../interdiff_data/temp')
    parser.add_argument('--save_db_path', type=str, default='../interdiff_data/pocket_prompt_test.lmdb')
    parser.add_argument('--radius', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.save_pocket_path, exist_ok=True)
    os.makedirs(args.temp_path, exist_ok=True)

    with open(os.path.join(args.source_data_path, 'index.pkl'), 'rb') as f:
        index = pickle.load(f)
    db = lmdb.open(
        args.save_db_path,
        map_size=int(10 * (1024 * 1024 * 1024)),  # 10GB
        create=False,
        subdir=False,
        readonly=False,  # Writable
        lock=False,
        readahead=False,
        meminit=False,
    )
    txn = db.begin(write=True, buffers=True)

    pool = mp.Pool(args.num_workers)
    index_pocket = []
    for i, item_pockets in enumerate(tqdm(pool.imap_unordered(partial(process_item, args=args), enumerate(index)), total=len(index))):
        item_pocket, protein_ligand_dicts = item_pockets
        data = {}
        if protein_ligand_dicts:
            index_pocket.append(item_pocket)

            # added to lmdb
            ligand_dict, selected_residues, prompt = protein_ligand_dicts
            pocket_dict = PDBProtein.residues_to_dict_atom_(selected_residues)

            data['protein_filename'] = item_pocket[0]
            data['ligand_filename'] = item_pocket[1]

            if pocket_dict is not None:
                for key, value in pocket_dict.items():
                    data['protein_' + key] = value

            if ligand_dict is not None:
                for key, value in ligand_dict.items():
                    data['ligand_' + key] = value

            # add the interaction prompt to every protein atom
            interactions = np.zeros(len(data['protein_element']), dtype=np.int64)
            for res_id_orig, interactions_type in prompt.items():
                interactions = np.where(data['protein_res_id_orig'] == res_id_orig, interactions_type, interactions)
            data['interactions'] = interactions

            # save
            txn.put(
                key=str(i).encode(),
                value=pickle.dumps(data)
            )

    txn.commit()
    pool.close()

    index_path = os.path.join(args.save_pocket_path, 'index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(index_pocket, f)
    print('Done. %d protein-ligand pairs in total.' % len(index_pocket))

    # split train/test data
    with db.begin() as txn:
        keys = list(txn.cursor().iternext(values=False))

    train = random.sample(range(len(keys)), int(0.998 * len(keys)))
    test = list(set(range(len(keys))).difference(set(train)))
    # train nums: 165640 , test nums 332
    print('train nums: {} , test nums {}'.format(len(train), len(test)))
    torch.save({'train': train, 'test': test}, '../interdiff_data/prompt_split.pt')

    db.close()
