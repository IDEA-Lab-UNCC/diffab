import os
import argparse
import ray
import shelve
import time
import pandas as pd
from typing import Mapping

from diffab.tools.eval.base import EvalTask, TaskScanner, evaluation_dir, run_dir
from diffab.tools.eval.similarity import eval_similarity


@ray.remote(num_cpus=1)
def evaluate(task, args):
    funcs = []
    funcs.append(eval_similarity)
    if not args.no_energy:
        # Imported lazily: `energy` calls pyrosetta.init() at module scope, so a
        # top-level import would make the whole package require PyRosetta even
        # when energy evaluation is switched off.
        from diffab.tools.eval.energy import eval_interface_energy
        funcs.append(eval_interface_energy)
    for f in funcs:
        task = f(task)
    return task


def dump_db(db: Mapping[str, EvalTask], path):
    """Write one run's `summary.csv` from that run's database."""
    table = []
    for task in db.values():
        if 'abopt' in path and task.scores['seqid'] >= 100.0:
            # In abopt (Antibody Optimization) mode, ignore sequences identical to the wild-type
            continue
        table.append(task.to_report_dict())
    table = pd.DataFrame(table)
    table.to_csv(path, index=False, float_format='%.6f')
    return table


class RunDatabases:
    """One shelve database per run, inside that run's evaluation/ directory.

    The evaluation output lives with the run it describes, so the cache does too
    -- deleting a run takes its cache with it, rather than leaving orphaned
    entries in a tree-wide database.
    """

    def __init__(self):
        self.dbs = {}

    def get(self, directory):
        if directory not in self.dbs:
            path = os.path.join(evaluation_dir(directory), 'evaluation_db')
            self.dbs[directory] = shelve.open(path)
        return self.dbs[directory]

    def already_done(self, task):
        return task.in_path in self.get(run_dir(task))

    def save(self, task):
        task.save_to_db(self.get(run_dir(task)))

    def dump(self):
        for directory, db in self.dbs.items():
            db.sync()
            dump_db(db, os.path.join(evaluation_dir(directory), 'summary.csv'))

    def close(self):
        for db in self.dbs.values():
            db.close()
        self.dbs.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./results')
    parser.add_argument('--pfx', type=str, default='rosetta')
    parser.add_argument('--no_energy', action='store_true', default=False)
    parser.add_argument('--once', action='store_true', default=False,
                        help='stop after one pass instead of watching for new results')
    args = parser.parse_args()
    ray.init()

    databases = RunDatabases()
    # db=None: the scanner tracks what it has already handed out within this
    # process, and each run's own database decides what was scored previously.
    scanner = TaskScanner(root=args.root, postfix=args.pfx)
    try:
        while True:
            tasks = [t for t in scanner.scan() if not databases.already_done(t)]
            futures = [evaluate.remote(t, args) for t in tasks]
            if len(futures) > 0:
                print(f'Submitted {len(futures)} tasks.')
            while len(futures) > 0:
                done_ids, futures = ray.wait(futures, num_returns=1)
                for done_id in done_ids:
                    done_task = ray.get(done_id)
                    databases.save(done_task)
                    print(f'Remaining {len(futures)}. Finished {done_task.in_path}')

            databases.dump()
            if args.once:
                break
            time.sleep(1.0)
    finally:
        databases.close()


if __name__ == '__main__':
    main()
