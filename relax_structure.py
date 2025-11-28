#! /usr/bin/env python
"""Post-hoc relax script after AlphaFold prediction."""

import json
import os

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.relax import relax


logging.set_verbosity(logging.INFO)

flags.DEFINE_string('input_pdb', None, 'Path to the input PDB file to relax')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')
flags.DEFINE_boolean(
    'use_gpu_relax',
    None,
    'Whether to relax on GPU. Relax on GPU can be much faster than CPU, so it is recommended to enable if possible. '
    'GPUs must be available if this setting is enabled.'
)
flags.DEFINE_string("status_file", None, 'Status report file path.')


FLAGS = flags.FLAGS

RELAX_MAX_ITERATIONS = 0
RELAX_ENERGY_TOLERANCE = 2.39
RELAX_STIFFNESS = 10.0
RELAX_EXCLUDE_RESIDUES = []
RELAX_MAX_OUTER_ITERATIONS = 3


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  amber_relaxer = relax.AmberRelaxation(
    max_iterations=RELAX_MAX_ITERATIONS,
    tolerance=RELAX_ENERGY_TOLERANCE,
    stiffness=RELAX_STIFFNESS,
    exclude_residues=RELAX_EXCLUDE_RESIDUES,
    max_outer_iterations=RELAX_MAX_OUTER_ITERATIONS,
    use_gpu=FLAGS.use_gpu_relax)

  logging.info('Relaxing %s', FLAGS.input_pdb)
  model_name = os.path.basename(FLAGS.input_pdb).split('.pdb')[0]

  with open(FLAGS.input_pdb, 'r') as fh:
      structure = protein.from_pdb_string(fh.read())

  relaxed_pdb_str, _, violations = amber_relaxer.process(prot=structure)
  relax_metrics= {
    'remaining_violations': violations,
    'remaining_violations_count': sum(violations)
  }

  relaxed_output_path = os.path.join(FLAGS.output_dir, f'relaxed_{model_name}.pdb')
  with open(relaxed_output_path, 'w') as f:
    f.write(relaxed_pdb_str)

  relax_metrics_path = os.path.join(FLAGS.output_dir, 'relax_metrics.json')
  with open(relax_metrics_path, 'w') as f:
    f.write(json.dumps(relax_metrics, indent=4))


if __name__ == '__main__':
  flags.mark_flags_as_required(['input_pdb', 'output_dir', 'use_gpu_relax'])

  app.run(main)
