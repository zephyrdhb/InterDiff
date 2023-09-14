import os
import argparse
import multiprocessing as mp
from multiprocessing import set_start_method
import pickle
import shutil
from functools import partial
import lmdb
from tqdm.auto import tqdm
from utils.data import PDBProtein, parse_sdf_file


def load_item(item, path):
    pdb_path = os.path.join(path, item[0])
    sdf_path = os.path.join(path, item[1])
    with open(pdb_path, 'r') as f:
        pdb_block = f.read()
    with open(sdf_path, 'r') as f:
        sdf_block = f.read()
    return pdb_block, sdf_block


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

        return (pocket_fn, ligand_fn, item[0], item[2]), (ligand_dict, selected_residues)  # item[0]: original protein filename; item[2]: rmsd.
    except Exception as e:
        print(e)
        return (None, item[1], item[0], item[2]), None


if __name__ == '__main__':
    set_start_method('fork')
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data_path', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--save_pocket_path', type=str, default='../interdiff_data/crossdocked_v1.1_rmsd1.0_pocket')
    parser.add_argument('--save_db_path', type=str, default='../interdiff_data/pocket.lmdb')
    parser.add_argument('--radius', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    args = parser.parse_args()

    os.makedirs(args.save_pocket_path, exist_ok=True)
    with open(os.path.join(args.source_data_path, 'index.pkl'), 'rb') as f:
        index = pickle.load(f)
    db = lmdb.open(
        args.save_db_path,
        map_size=10 * (1024 * 1024 * 1024),  # 10GB
        create=False,
        subdir=False,
        readonly=False,  # Writable
        lock=False,
        readahead=False,
        meminit=False,
    )

    pool = mp.Pool(args.num_workers)
    index_pocket = []
    for i, item_pockets in enumerate(tqdm(pool.imap_unordered(partial(process_item, args=args), enumerate(index)), total=len(index))):
        item_pocket, protein_ligand_dicts = item_pockets
        if protein_ligand_dicts:
            index_pocket.append(item_pocket)

            # added to lmdb
            ligand_dict, selected_residues = protein_ligand_dicts
            pocket_dict = PDBProtein.residues_to_dict_atom_(selected_residues)

            with db.begin(write=True, buffers=True) as txn:
                data = {}
                data['protein_filename'] = item_pocket[0]
                data['ligand_filename'] = item_pocket[1]

                if pocket_dict is not None:
                    for key, value in pocket_dict.items():
                        data['protein_' + key] = value

                if ligand_dict is not None:
                    for key, value in ligand_dict.items():
                        data['ligand_' + key] = value

                txn.put(
                    key=str(i).encode(),
                    value=pickle.dumps(data)
                )

    db.close()

    # index_pocket = pool.map(partial(process_item, args=args), index)
    pool.close()

    index_path = os.path.join(args.save_pocket_path, 'index.pkl')
    with open(index_path, 'wb') as f:
        pickle.dump(index_pocket, f)

    print('Done. %d protein-ligand pairs in total.' % len(index))
